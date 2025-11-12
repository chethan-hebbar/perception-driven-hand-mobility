# gemini_touch_task.py
import pybullet as p
import numpy as np
from src.simulation import Simulation
from src.robot_controller import RobotController
from src.gemini_perception import GeminiPerception
from src.prompts import PROMPT_V4_GEMINI_TOUCH

def main():
    """
    Main function to run the simplified "perceive and touch" task using only Gemini.
    """
    print("--- Initializing Gemini-Powered 'Touch' Task ---")

    sim = Simulation()
    controller = RobotController(sim.robot_id)
    try:
        perception_and_planning_module = GeminiPerception()
    except ValueError as e:
        print(f"CRITICAL ERROR: {e}. Is your .env file set up correctly?")
        return

    start_pos_actual = generate_reachable_point(controller)
    print(f"\nPlacing object at actual position: {[round(c, 3) for c in start_pos_actual]}")
    obj_id = sim.load_object("cube.urdf", start_pos_actual, global_scaling=0.1)
    p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1])
    sim.run(1)

    print("\n--- STAGE 1: PERCEIVE AND PLAN ---")
    rgb_image, _ = sim.get_camera_images()

    json_response = perception_and_planning_module.find_object_position(rgb_image, PROMPT_V4_GEMINI_TOUCH)
    
    if json_response and "position" in json_response:
        target_position = json_response["position"]
        print(f"✅ Gemini has identified the target position as: {target_position}")
    else:
        print("\n❌ Gemini failed to identify a valid target position. Aborting.")
        return

    print("\n--- STAGE 2: EXECUTION ---")
    print(f"Commanding robot arm to move to the target position...")
    
    controller.move_to(target_position)
            
    print("\n>>> TASK COMPLETE: Arm has moved to the perceived location. <<<")
    
    p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=-1, basePosition=target_position)

    sim.run(1)

def generate_reachable_point(controller):
    """Generates a random point and checks if it's reachable by the robot arm."""
    min_x, max_x = 0.3, 0.7; min_y, max_y = -0.4, 0.4; z = 0.05
    while True:
        pos = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y), z]
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_poses = p.calculateInverseKinematics(controller.robot_id, controller.end_effector_link_index, pos, target_orn)
        if joint_poses is not None: return pos

if __name__ == '__main__':
    main()