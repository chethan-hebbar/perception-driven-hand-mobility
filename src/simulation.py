import pybullet as p
import pybullet_data
import time
import numpy as np

class Simulation:
    """Handles the PyBullet simulation setup and camera control."""
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        
        self.camera_width = 224
        self.camera_height = 224
        self.view_matrix = None
        self.projection_matrix = None
        
        self._setup_camera()
        print("Simulation initialized.")

    def _setup_camera(self):
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1, 0, 1],
            cameraTargetPosition=[0.4, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=0.1,
            farVal=3.1
        )

    def get_camera_images(self):
        width, height, rgb, depth, seg = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix
        )
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        depth_array = np.array(depth).reshape((height, width))
        far, near = 3.1, 0.1
        depth_m = far * near / (far - (far - near) * depth_array)
        return rgb_array, depth_m

    def load_object(self, urdf_path, position, orientation=[0,0,0,1], global_scaling=1.0):
        obj_id = p.loadURDF(urdf_path, position, orientation, globalScaling=global_scaling)
        return obj_id

    def run(self, duration_sec):
        for _ in range(int(duration_sec * 240)):
            p.stepSimulation()
            time.sleep(1./240.)
            
    def __del__(self):
        p.disconnect()