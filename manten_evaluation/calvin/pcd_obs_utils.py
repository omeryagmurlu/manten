import numpy as np
import pybullet as pb


def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a pixel point to 3D coordinates
    Args
        point: tuple (u, v); pixel coordinates of point to deproject
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output
        (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix).reshape((4, 4)).T)
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.
    if sanity_check:
        sample_inds = np.random.permutation(u.shape[0])[:2000]  # noqa: NPY002
        for ind in sample_inds:
            cam_world_pos = cam.deproject((u[ind], v[ind]), depth_img, homogeneous=True)
            assert np.abs(cam_world_pos - world_pos[:, ind]).max() <= 1e-3  # noqa: PLR2004

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos


def get_gripper_camera_view_matrix(cam):
    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid, linkIndex=cam.gripper_cam_link, physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = pb.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
    return view_matrix


def compute_pcd_as_part_of_obs(obs, env):
    """
    Compute the point cloud of the observation using the depth images and the camera intrinsics of a PlayTableEnv.
    """

    depth_static = obs["depth_obs"]["depth_static"]
    depth_gripper = obs["depth_obs"]["depth_gripper"]

    static_cam = env.cameras[0]
    gripper_cam = env.cameras[1]
    gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)

    static_pcd = deproject(
        static_cam, depth_static, homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    static_pcd = np.reshape(static_pcd, (depth_static.shape[0], depth_static.shape[1], 3))
    gripper_pcd = deproject(
        gripper_cam, depth_gripper, homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    gripper_pcd = np.reshape(gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3))

    obs["pcd_obs"] = {}
    obs["pcd_obs"]["pcd_static"] = static_pcd
    obs["pcd_obs"]["pcd_gripper"] = gripper_pcd

    return obs
