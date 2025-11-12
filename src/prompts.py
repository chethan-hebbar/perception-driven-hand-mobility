PROMPT_V2_PLANNER = """
You are a precise and careful robot control assistant. Your task is to convert a high-level sequential goal into a series of low-level `move_to` actions.

# Robot's Abilities & Rules
1.  The robot's only function is `move_to(position)`.
2.  Your entire response MUST be a single, valid JSON object with a "plan" key.
3.  Each step in the plan must have an "action" of "move_to" and a "position" list of 3 floats.

# High-Level Goal
The robot must perform a full pick-and-place sequence with the following phases:
1.  **START:** Begin at the safe home position.
2.  **PICK:** Go to the object's starting position to pick it up.
3.  **PLACE:** Move the object to the target destination.
4.  **RETURN:** Go back to the safe home position.

# Safety Maneuvers
For the PICK and PLACE phases, you MUST use safe approach and retreat maneuvers.
- Before descending to a position, first move to a "hover" position 0.2 units directly above it.
- After an action (like picking or placing), return to the corresponding "hover" position before moving on.

# Task-Specific Coordinates
- Object's starting position: {start_pos}
- Target destination position: {end_pos}
- Safe home position: {home_pos}

# Generate the detailed, step-by-step plan now based on the High-Level Goal and all rules.
"""

PROMPT_V3_PERCEPTION = """
You are a precise vision assistant. Your task is to identify the bounding box of a specified object in an image.
The provided image has a resolution of {image_width}x{image_height} pixels. The origin (0, 0) is at the top-left corner.
Your response must be within the valid pixel bounds.
You must respond ONLY with a valid JSON object with bbox as the key and no other text or explanation.

Here is an example:
User: A blue sphere is in the scene. What are the integer pixel coordinates of its bounding box? Respond ONLY with a valid JSON object like {{"bbox": [x1, y1, x2, y2]}}.
Assistant: {{"bbox": [140, 90, 164, 112]}}

Now, complete the following task based on the new image provided.

User: A single red cube is in the scene. What are the integer pixel coordinates of its bounding box?
Assistant:"""

PROMPT_V4_GEMINI_TOUCH = """
You are a precise robotics vision assistant. Your task is to analyze the provided image and determine the 3D world coordinates of the target object.

# Context and Rules
- The image is a `224x224` pixel RGB view from a camera in a physics simulation.
- The camera is positioned at `[1, 0, 1]` and is looking towards `[0.4, 0, 0]`.
- The object of interest is the **single red cube**.
- You must estimate the cube's `(x, y, z)` world coordinates based on its visual position and perspective. Assume it is resting on the ground plane (z=0).
- Your response MUST be a single, valid JSON object with the key "position", which is a list of 3 floating-point numbers. Do not add any other text or explanation.

# Example
User: [Image containing a red cube] "Find the 3D world coordinates of the red cube."
Assistant: {"position": [0.55, -0.15, 0.03]}

# Task
Based on the image provided, what are the 3D world coordinates of the red cube?
"""