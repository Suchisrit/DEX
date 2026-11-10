import torch


def invert_theta(rho, D, newton_iters=3, bisect_iters=10, eps=1e-8, theta_max=1.5533):
    '''
    Inverts the Kannala-Brandt distortion polynomial theta_d = theta * (1 + k1 theta^2 + ... + k4 theta^8)

    Arg(s):
        rho : torch.Tensor[float32]
            B x 1 x H x W distorted radius theta_d (radians) of the normalized image coordinates
        D : torch.Tensor[float64]
            B x 4 distortion coefficients [k1, k2, k3, k4]
        newton_iters : int
            Newton iterations
        bisect_iters : int
            bisection iterations used where Newton has not converged
        eps : float
            numerical stability constant
        theta_max : float
            largest incidence angle (radians) considered (~89 degrees)
    Returns:
        torch.Tensor : B x 1 x H x W undistorted incidence angle theta
    '''

    B = rho.shape[0]
    k1, k2, k3, k4 = [D[:, i].reshape(B, 1, 1, 1) for i in range(4)]

    def theta2thetad(th):
        t2 = th * th
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        return th * (1 + k1*t2 + k2*t4 + k3*t6 + k4*t8)

    zeros = torch.zeros_like(rho)
    theta_max_tensor = torch.full_like(rho, theta_max)

    rho_max = theta2thetad(theta_max_tensor)
    rho_clamped = torch.clamp(rho, min=zeros, max=rho_max)

    # Newton warm start (theta ~ rho for small angles)
    theta = torch.clamp(rho_clamped, min=zeros, max=theta_max_tensor).clone()
    for _ in range(newton_iters):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        P = 1 + k1*t2 + k2*t4 + k3*t6 + k4*t8
        f = theta * P - rho_clamped
        fp = 1 + 3*k1*t2 + 5*k2*t4 + 7*k3*t6 + 9*k4*t8
        theta = torch.clamp(theta - f / (fp + eps), min=zeros, max=theta_max_tensor)

    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    f = theta * (1 + k1*t2 + k2*t4 + k3*t6 + k4*t8) - rho_clamped
    need_bisect = (f.abs() > 1e-6)

    if need_bisect.any():
        lo = torch.zeros_like(theta)
        hi = theta_max_tensor.clone()

        Flo = theta2thetad(lo) - rho_clamped
        Fhi = theta2thetad(hi) - rho_clamped
        bad = (Flo > 0) | (Fhi < 0)

        th = theta.clone()
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            Fmid = theta2thetad(mid) - rho_clamped
            go_hi = (Fmid < 0)
            lo = torch.where(go_hi, mid, lo)
            hi = torch.where(go_hi, hi, mid)
            th = mid

        theta = torch.where(need_bisect & ~bad, th, theta)

    # exact center
    theta = torch.where(rho > eps, theta, torch.zeros_like(theta))
    return theta


def fisheye_rays(K, D, shape, device, dtype):
    '''
    Computes the undistorted (pinhole) viewing ray of every pixel of a Kannala-Brandt fisheye camera

    Arg(s):
        K : torch.Tensor[float]
            B x 3 x 3 intrinsics
        D : torch.Tensor[float]
            B x 4 fisheye distortion coefficients
        shape : tuple[int, int]
            H, W of the image
        device : torch.device
            device to create the pixel grid on
        dtype : torch.dtype
            dtype of the pixel grid
    Returns:
        torch.Tensor : B x 3 x H x W rays (x/z, y/z, 1)
    '''

    B = K.shape[0]
    H, W = shape

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    Kinv = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    Kinv[:, 0, 0] = 1.0/fx
    Kinv[:, 1, 1] = 1.0/fy
    Kinv[:, 0, 2] = -cx/fx
    Kinv[:, 1, 2] = -cy/fy
    Kinv[:, 2, 2] = 1.0

    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(1, 1, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(1, 1, H, W)
    ones = torch.ones((1, 1, H, W), device=device, dtype=dtype)
    p = torch.cat([u, v, ones], dim=1)
    p = p.expand(B, -1, -1, -1).contiguous()

    p_camera = torch.einsum('bij,bjhw->bihw', Kinv, p)

    # undo the fisheye distortion on the normalized coordinates
    xprime = p_camera[:, 0:1]
    yprime = p_camera[:, 1:2]
    rho = torch.sqrt(xprime*xprime + yprime*yprime)
    theta = invert_theta(rho, D)
    r = torch.tan(theta)
    scale = torch.where(rho > 1e-8, r / rho, torch.ones_like(r))
    p_camera[:, 0:2, :, :] = p_camera[:, 0:2, :, :] * scale

    return p_camera


def fisheye_distance_to_fisheye_depth(R, K, D):
    '''
    Converts Euclidean distance from the camera center to z-depth for a Kannala-Brandt fisheye camera

    Arg(s):
        R : torch.Tensor[float32]
            B x 1 x H x W distance in meters
        K : torch.Tensor[float]
            B x 3 x 3 intrinsics
        D : torch.Tensor[float]
            B x 4 fisheye distortion coefficients
    Returns:
        torch.Tensor[float32] : B x 1 x H x W depth in meters
    '''

    p_camera = fisheye_rays(K, D, R.shape[2:], R.device, R.dtype)

    p_camera_norm = torch.norm(p_camera, dim=1, keepdim=True)
    scale_factor = R / p_camera_norm
    p_backproj = p_camera * scale_factor.expand(-1, 3, -1, -1)

    return p_backproj[:, 2:3, :, :]


def fisheye_depth_to_fisheye_distance(d, K, D):
    '''
    Converts z-depth to Euclidean distance from the camera center for a Kannala-Brandt fisheye camera

    Arg(s):
        d : torch.Tensor[float32]
            B x 1 x H x W depth in meters
        K : torch.Tensor[float]
            B x 3 x 3 intrinsics
        D : torch.Tensor[float]
            B x 4 fisheye distortion coefficients
    Returns:
        torch.Tensor[float32] : B x 1 x H x W distance in meters
    '''

    p_camera = fisheye_rays(K, D, d.shape[2:], d.device, d.dtype)

    return torch.norm(p_camera * d, dim=1, keepdim=True)
