import pybullet as p
import time
import math

class RobotController:
    """Handles the low-level control of the robot arm."""
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.num_joints = p.getNumJoints(self.robot_id)
        self.end_effector_link_index = self._find_end_effector()
        self.movable_joint_indices = self._find_movable_joints()
        
        print(f"RobotController initialized for robot ID: {self.robot_id}")
        print(f"Found end-effector at link index: {self.end_effector_link_index}")
        print(f"Found {len(self.movable_joint_indices)} movable joints.")

    def _find_end_effector(self):
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[12].decode('UTF-8') == 'panda_hand':
                return i
        return 8 # Fallback

    def _find_movable_joints(self):
        indices = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE or info[2] == p.JOINT_PRISMATIC:
                indices.append(i)
        return indices

    def move_to(self, target_pos, target_orn=None, duration_sec=2):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        joint_poses = p.calculateInverseKinematics(
            self.robot_id, self.end_effector_link_index, target_pos, target_orn
        )
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.movable_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses[:len(self.movable_joint_indices)],
            forces=[500] * len(self.movable_joint_indices)
        )
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            p.stepSimulation()
            time.sleep(1./240.)