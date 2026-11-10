"""Renders the ScanNet++ DSLR ground-truth depth for the images in splits/scannetpp_test.txt

Equivalent to common/render.py of the ScanNet++ toolkit (https://github.com/scannetpp/scannetpp, which uses renderpy)
but renders only the split images and needs no OpenGL: the renderer is implemented in PyTorch, see fisheye_mesh_depth.py.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fisheye_mesh_depth import load_ply, render_depth


def qvec2rotmat(q):
    return np.array([
        [1 - 2 * q[2]**2 - 2 * q[3]**2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[3] * q[1] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1]**2 - 2 * q[3]**2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[3] * q[1] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1]**2 - 2 * q[2]**2]])


def read_colmap_text(colmap_dir):
    '''Returns the single camera (model, width, height, params) and the world-to-camera pose of every image'''

    with open(os.path.join(colmap_dir, 'cameras.txt'), 'r') as f:
        lines = [line.split() for line in f if not line.startswith('#') and line.strip()]
    assert len(lines) == 1, 'Multiple cameras not supported'
    camera = (lines[0][1], int(lines[0][2]), int(lines[0][3]), np.array([float(p) for p in lines[0][4:]]))

    poses = {}
    with open(os.path.join(colmap_dir, 'images.txt'), 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
    # two lines per image: pose, then 2D points
    for line in lines[0::2]:
        elems = line.split()
        if not elems:
            continue
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = qvec2rotmat(np.array([float(e) for e in elems[1:5]]))
        world_to_camera[:3, 3] = np.array([float(e) for e in elems[5:8]])
        poses[elems[9]] = world_to_camera

    return camera, poses


def main():
    parser = argparse.ArgumentParser(description='Render ScanNet++ DSLR depth for the evaluation split')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Downloaded ScanNet++ root containing data/<scene>/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output root, depth is saved to <output_dir>/<scene>/dslr/render_depth/')
    parser.add_argument('--split_path', type=str, default='splits/scannetpp_test.txt')
    parser.add_argument('--near', type=float, default=0.05)
    parser.add_argument('--far', type=float, default=20.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    images_per_scene = defaultdict(list)
    with open(args.split_path, 'r') as f:
        for line in f:
            image, depth = line.split(' ')[:2]
            images_per_scene[depth.split('/')[1]].append(os.path.basename(image))

    for scene_id, image_names in tqdm(images_per_scene.items(), desc='scene'):
        scene_dir = os.path.join(args.data_root, 'data', scene_id)
        (model, width, height, params), poses = read_colmap_text(os.path.join(scene_dir, 'dslr', 'colmap'))
        assert model == 'OPENCV_FISHEYE', f'Unsupported camera model {model}'

        fx, fy, cx, cy, k1, k2, k3, k4 = params
        camera = (width, height, fx, fy, cx, cy, k1, k2, k3, k4)

        vertices, faces = load_ply(os.path.join(scene_dir, 'scans', 'mesh_aligned_0.05.ply'))
        vertices = torch.from_numpy(vertices).to(device)
        faces = torch.from_numpy(faces).long().to(device)

        depth_dir = os.path.join(args.output_dir, scene_id, 'dslr', 'render_depth')
        os.makedirs(depth_dir, exist_ok=True)

        for image_name in image_names:
            world_to_camera = torch.from_numpy(poses[image_name]).float().to(device)
            depth = render_depth(vertices, faces, world_to_camera, camera, args.near, args.far)
            # depth in millimeters, 0 is invalid
            depth = (depth.cpu().numpy() * 1000).clip(0, 65535).astype(np.uint16)
            Image.fromarray(depth).save(os.path.join(depth_dir, os.path.splitext(image_name)[0] + '.png'))


if __name__ == '__main__':
    main()
