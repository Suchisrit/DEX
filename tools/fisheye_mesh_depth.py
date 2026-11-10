"""Pure PyTorch re-implementation of the renderpy fisheye depth renderer used by the ScanNet++ toolkit

renderpy (https://github.com/liu115/renderpy) rasterizes the mesh with OpenGL. For OPENCV_FISHEYE cameras its vertex
shader applies the fisheye distortion to every vertex and the triangles are then rasterized with a pinhole projection.
The depth of a pixel is the perspective-correct interpolation of the camera-space z of the triangle vertices. This
module reproduces that pipeline: vertex distortion, near/far clipping, fixed point (1/256 pixel) rasterization with the
top-left fill rule, and z-buffering.
"""

import numpy as np
import torch

SUBPIXEL = 256
# renderpy clamps the (infinite) radius cutoff of the fisheye camera to 1e3
RADIUS_CUTOFF_SQUARED = 1e3


def load_ply(path):
    '''Loads the vertices and triangles of a binary little endian PLY mesh (x y z float, r g b uchar, uchar-int faces)'''

    with open(path, 'rb') as f:
        n_vertices = n_faces = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line == 'end_header':
                break
        data = f.read()

    vertex_dtype = np.dtype([('xyz', '<f4', 3), ('rgb', 'u1', 3)])
    face_dtype = np.dtype([('n', 'u1'), ('idx', '<i4', 3)])
    vertex_bytes = n_vertices * vertex_dtype.itemsize

    vertices = np.frombuffer(data, dtype=vertex_dtype, count=n_vertices)['xyz']
    faces = np.frombuffer(data, dtype=face_dtype, count=n_faces, offset=vertex_bytes)
    assert (faces['n'] == 3).all(), 'Only triangle meshes are supported'

    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces['idx'])


def distort_vertices(points, fx, fy, cx, cy, distortion, world_to_camera):
    '''Vertex shader of renderpy for OPENCV_FISHEYE: returns the distorted camera coordinates (x', y', z)'''

    k1, k2, k3, k4 = distortion
    R = world_to_camera[:3, :3].to(points.dtype)
    t = world_to_camera[:3, 3].to(points.dtype)
    cam = points @ R.t() + t

    z = cam[:, 2]
    nx = cam[:, 0] / z
    ny = cam[:, 1] / z
    r2 = nx * nx + ny * ny

    inside = r2 < RADIUS_CUTOFF_SQUARED
    r = torch.sqrt(r2)
    scaled = inside & (r > 1e-6)
    theta_by_r = torch.where(scaled, torch.atan(r) / r.clamp_min(1e-12), torch.ones_like(r))
    nx = nx * theta_by_r
    ny = ny * theta_by_r
    r2 = torch.where(scaled, theta_by_r * theta_by_r * r2, r2)

    factor = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    factor = torch.where(inside, factor, torch.full_like(factor, 99.0))

    return torch.stack([z * factor * nx, z * factor * ny, z], dim=1)


def clip_plane(tris, dist):
    '''
    Clips triangles against a plane

    Arg(s):
        tris : torch.Tensor[float]
            K x 3 x 3 vertex attributes (x', y', z)
        dist : torch.Tensor[float]
            K x 3 signed distance of every vertex to the plane, non-negative is inside
    Returns:
        torch.Tensor[float] : K' x 3 x 3 clipped triangles
    '''

    inside = dist >= 0
    n_in = inside.sum(1)
    out = [tris[n_in == 3]]

    def lerp(a, b, da, db):
        t = (da / (da - db)).unsqueeze(1)
        return a + t * (b - a)

    # one vertex inside: keep the triangle formed by it and the two edge intersections
    sel = n_in == 1
    if sel.any():
        t, d, ins = tris[sel], dist[sel], inside[sel]
        i = ins.float().argmax(1)
        ar = torch.arange(len(i), device=i.device)
        j, k = (i + 1) % 3, (i + 2) % 3
        pi, pj, pk = t[ar, i], t[ar, j], t[ar, k]
        di, dj, dk = d[ar, i], d[ar, j], d[ar, k]
        out.append(torch.stack([pi, lerp(pi, pj, di, dj), lerp(pi, pk, di, dk)], dim=1))

    # two vertices inside: the clipped polygon is a quad, split it into two triangles
    sel = n_in == 2
    if sel.any():
        t, d, ins = tris[sel], dist[sel], inside[sel]
        o = (~ins).float().argmax(1)
        ar = torch.arange(len(o), device=o.device)
        p, q = (o + 1) % 3, (o + 2) % 3
        po, pp, pq = t[ar, o], t[ar, p], t[ar, q]
        do, dp, dq = d[ar, o], d[ar, p], d[ar, q]
        i_op = lerp(pp, po, dp, do)
        i_qo = lerp(pq, po, dq, do)
        out.append(torch.stack([pp, pq, i_qo], dim=1))
        out.append(torch.stack([pp, i_qo, i_op], dim=1))

    return torch.cat(out, dim=0)


def rasterize_depth(X, Y, Z, height, width, fragment_budget=1 << 23):
    '''
    Rasterizes triangles into a depth buffer

    Arg(s):
        X, Y : torch.Tensor[int64]
            T x 3 window coordinates of the vertices in 1/256 pixels
        Z : torch.Tensor[float64]
            T x 3 camera z of the vertices
        height, width : int
            size of the buffer
        fragment_budget : int
            maximum number of fragments evaluated at once
    Returns:
        torch.Tensor[float32] : height x width depth, 0 where no triangle covers the pixel
    '''

    device = X.device
    zbuf = torch.full((height * width,), float('inf'), dtype=torch.float32, device=device)

    # window coordinates must be non-degenerate and positively oriented
    area2 = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (Y[:, 1] - Y[:, 0]) * (X[:, 2] - X[:, 0])
    keep = area2 != 0
    X, Y, Z, area2 = X[keep], Y[keep], Z[keep], area2[keep]
    flip = area2 < 0
    order = torch.tensor([0, 1, 2], device=device).repeat(len(X), 1)
    order[flip] = torch.tensor([0, 2, 1], device=device)
    X, Y, Z = X.gather(1, order), Y.gather(1, order), Z.gather(1, order)
    area2 = area2.abs()

    half = SUBPIXEL // 2
    i_min = -((-(X.min(1).values - half)) // SUBPIXEL)
    i_max = (X.max(1).values - half) // SUBPIXEL
    j_min = -((-(Y.min(1).values - half)) // SUBPIXEL)
    j_max = (Y.max(1).values - half) // SUBPIXEL
    i_min, j_min = i_min.clamp_min(0), j_min.clamp_min(0)
    i_max, j_max = i_max.clamp_max(width - 1), j_max.clamp_max(height - 1)

    valid = (i_min <= i_max) & (j_min <= j_max)
    X, Y, Z, area2 = X[valid], Y[valid], Z[valid], area2[valid]
    i_min, i_max, j_min, j_max = i_min[valid], i_max[valid], j_min[valid], j_max[valid]

    size = torch.maximum(i_max - i_min + 1, j_max - j_min + 1)
    bucket = torch.ceil(torch.log2(size.float().clamp_min(4))).long()

    for b in torch.unique(bucket).tolist():
        side = 1 << b
        members = (bucket == b).nonzero().squeeze(1)
        chunk = max(1, fragment_budget // (side * side))
        offsets = torch.arange(side, device=device)

        for start in range(0, len(members), chunk):
            m = members[start:start + chunk]
            x, y, z, a2 = X[m], Y[m], Z[m], area2[m]
            px = i_min[m][:, None, None] + offsets[None, None, :]
            py = j_min[m][:, None, None] + offsets[None, :, None]
            in_box = (px <= i_max[m][:, None, None]) & (py <= j_max[m][:, None, None])
            cx = px * SUBPIXEL + half
            cy = py * SUBPIXEL + half

            covered = in_box
            edges = []
            for a, c in ((1, 2), (2, 0), (0, 1)):
                ax, ay = x[:, a][:, None, None], y[:, a][:, None, None]
                dx = (x[:, c] - x[:, a])[:, None, None]
                dy = (y[:, c] - y[:, a])[:, None, None]
                e = dx * (cy - ay) - dy * (cx - ax)
                top_left = (dy < 0) | ((dy == 0) & (dx < 0))
                covered = covered & ((e > 0) | ((e == 0) & top_left))
                edges.append(e)

            if not covered.any():
                continue

            inv_z = sum(e[covered].double() / a2[:, None, None].expand_as(e)[covered].double() /
                        z[:, k][:, None, None].expand_as(e)[covered] for k, e in enumerate(edges))
            depth = (1.0 / inv_z).float()
            index = (py * width + px)[covered]
            zbuf.scatter_reduce_(0, index, depth, reduce='amin', include_self=True)

    zbuf[torch.isinf(zbuf)] = 0
    return zbuf.view(height, width)


@torch.no_grad()
def render_depth(vertices, faces, world_to_camera, camera, near=0.05, far=20.0):
    '''
    Renders the z-depth of a mesh seen by an OPENCV_FISHEYE camera

    Arg(s):
        vertices : torch.Tensor[float32]
            N x 3 mesh vertices on the GPU
        faces : torch.Tensor[int64]
            M x 3 triangle indices on the GPU
        world_to_camera : torch.Tensor[float32]
            4 x 4 pose
        camera : tuple
            (width, height, fx, fy, cx, cy, k1, k2, k3, k4)
        near, far : float
            clipping planes in meters
    Returns:
        torch.Tensor[float32] : H x W depth in meters, 0 where the mesh is not visible
    '''

    width, height, fx, fy, cx, cy, *distortion = camera
    attributes = distort_vertices(vertices, fx, fy, cx, cy, distortion, world_to_camera)
    attributes = torch.nan_to_num(attributes, nan=0.0, posinf=0.0, neginf=0.0)
    tris = attributes[faces]
    z = tris[:, :, 2]

    # drop triangles outside of the depth range, clip the ones crossing a plane
    alive = (z >= near).any(1) & (z <= far).any(1)
    tris, z = tris[alive], z[alive]
    crossing = ~((z >= near) & (z <= far)).all(1)
    tris = torch.cat([tris[~crossing], clip_plane(tris[crossing], tris[crossing][:, :, 2] - near)], dim=0)
    crossing = ~(tris[:, :, 2] <= far).all(1)
    tris = torch.cat([tris[~crossing], clip_plane(tris[crossing], far - tris[crossing][:, :, 2])], dim=0)

    tris = tris.double()
    z = tris[:, :, 2]
    sx = fx * tris[:, :, 0] / z + cx + 0.5
    sy = fy * tris[:, :, 1] / z + cy + 0.5
    limit = 1e6
    X = torch.round(sx.clamp(-limit, limit) * SUBPIXEL).long()
    Y = torch.round(sy.clamp(-limit, limit) * SUBPIXEL).long()

    return rasterize_depth(X, Y, z, height, width)
