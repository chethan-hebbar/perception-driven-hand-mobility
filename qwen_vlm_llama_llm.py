import pybullet as p
import numpy as np
from src.simulation import Simulation
from src.robot_controller import RobotController
from src.vlm_perception import VLMPception
from src.llm_planner import LLMPlanner
from src.prompts import PROMPT_V2_PLANNER, PROMPT_V3_PERCEPTION
from src.gemini_perception import GeminiPerception  

def main():
    """
    Main function to run the complete, perception-driven robotics task.
    """
    print("--- Initializing Perception-Driven Robotic System ---")

    # --- 1. System Initialization ---
    sim = Simulation()
    controller = RobotController(sim.robot_id)
    perception = VLMPception(model='qwen3-vl:8b')
    planner = LLMPlanner(model='llama3:8b')

    # --- 2. Scene Setup and Perception ---
    start_pos_actual = generate_reachable_point(controller)
    
    print(f"\nPlacing object at reachable position: {[round(c, 2) for c in start_pos_actual]}")
    obj_id = sim.load_object("cube.urdf", start_pos_actual, global_scaling=0.1)
    p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1])
    sim.run(1)

    # --- 3. PERCEPTION STAGE (with retries) ---
    print("\n--- STAGE 1: PERCEPTION (Bounding Box Method) ---")
    start_pos_perceived = perceive_object_location(sim, perception)
    if start_pos_perceived is None:
        print("\n❌ Perception failed after all attempts. Shutting down.")
        return

    # --- 4. PLANNING & EXECUTION ---
    print("\n--- STAGE 2: PLANNING & EXECUTION ---")
    end_pos = [0.0, 0.5, 0.3]
    home_pos = [0.3, 0.0, 0.5]

    formatted_prompt = PROMPT_V2_PLANNER.format(
        start_pos=[round(c, 3) for c in start_pos_perceived],
        end_pos=end_pos,
        home_pos=home_pos
    )

    print("Asking Planner LLM to generate a motion plan...")
    json_plan = planner.generate_plan(formatted_prompt)

    if json_plan and "plan" in json_plan and isinstance(json_plan["plan"], list):
        plan = json_plan["plan"]
        print(f"✅ Plan received with {len(plan)} steps. Executing now...")
        
        for i, step in enumerate(plan):
            print(f"\n--- Executing Step {i+1}/{len(plan)} ---")
            position = step.get("position")
            if position:
                print(f"Action: move_to, Position: {position}")
                controller.move_to(position)
            else:
                print(f"Skipping malformed step: {step}")
        
        print("\n>>> FULL TASK COMPLETE <<<")
    else:
        print("\n--- Failed to generate a valid plan. Aborting. ---")

    sim.run(15)

def generate_reachable_point(controller):
    """Generates a random point and checks if it's reachable by the robot arm."""
    print("Generating a guaranteed reachable point for the object...")
    min_x, max_x = 0.3, 0.7
    min_y, max_y = -0.4, 0.4
    z = 0.025
    while True:
        pos = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y), z]
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_poses = p.calculateInverseKinematics(controller.robot_id, controller.end_effector_link_index, pos, target_orn)
        if joint_poses is not None and len(joint_poses) > 0:
            print(f"  -> Found reachable point: {[round(c, 2) for c in pos]}")
            return pos

def perceive_object_location(sim, perception_module, max_attempts=3):
    """
    Uses the VLM to perceive the object's location, with retries.
    Returns the 3D world coordinate or None if it fails.
    """
    for attempt in range(max_attempts):
        print(f"\nPerception Attempt {attempt + 1}/{max_attempts}...")
        rgb_image, depth_image_m = sim.get_camera_images()
        
        height, width, _ = rgb_image.shape
        question = PROMPT_V3_PERCEPTION.format(image_width=width, image_height=height)
        
        print("Asking VLM for the object's bounding box...")
        json_response = perception_module.find_object_position(rgb_image, question)
        
        if json_response and "bbox" in json_response and len(json_response["bbox"]) == 4:
            x1, y1, x2, y2 = json_response["bbox"]
            if (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height and x1 < x2 and y1 < y2):
                print(f"✅ VLM returned a valid bounding box: {[x1, y1, x2, y2]}")
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)
                print(f"Calculated center pixel: [{px}, {py}]")
                return deproject_pixel_to_world(sim, px, py, depth_image_m)
            else:
                print(f"⚠️ VLM returned an out-of-bounds bounding box. Retrying...")
        else:
            print(f"⚠️ VLM returned a malformed response. Retrying...")
    return None

def deproject_pixel_to_world(sim, px, py, depth_image_m):
    """Converts a 2D pixel coordinate to a 3D world coordinate."""
    depth_m = depth_image_m[py, px]
    x_ndc = (2 * px / sim.camera_width) - 1.0
    y_ndc = 1.0 - (2 * py / sim.camera_height)
    near, far = 0.1, 3.1
    z_ndc = (far + near) / (far - near) - (2 * far * near) / ((far - near) * depth_m)
    pos_clip = np.array([x_ndc, y_ndc, z_ndc, 1.0])

    proj_matrix = np.asarray(sim.projection_matrix).reshape([4, 4], order="F")
    inv_proj_matrix = np.linalg.inv(proj_matrix)
    pos_camera_homogeneous = np.dot(inv_proj_matrix, pos_clip)
    pos_camera = pos_camera_homogeneous[:3] / pos_camera_homogeneous[3]

    view_matrix = np.asarray(sim.view_matrix).reshape([4, 4], order="F")
    inv_view_matrix = np.linalg.inv(view_matrix)
    pos_camera_4d = np.append(pos_camera, 1.0)
    pos_world_homogeneous = np.dot(inv_view_matrix, pos_camera_4d)
    pos_world = pos_world_homogeneous[:3]
    
    print(f"✅ Successfully localized object at 3D position: {[round(c, 3) for c in pos_world]}")
    return pos_world

if __name__ == '__main__':
    main()