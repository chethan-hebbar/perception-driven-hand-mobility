# Perception-Driven Robotic Manipulation using Language Models

![Project Demo GIF]

This project is a comprehensive exploration into building an autonomous robotics system where a simulated robot arm can perceive its environment, formulate a plan, and execute a task based on instructions processed by Large Language Models (LLMs). The system is built in Python using the PyBullet physics simulator and leverages both local and API-based AI models for different parts of the intelligent agent loop.

The core of this repository is not just a single working implementation, but a comparative study of different state-of-the-art perception methodologies, demonstrating the trade-offs between speed, accuracy, and the "Sim-to-Real" domain gap.

---

## 🏛️ System Architecture

The final, successful architecture implements a modular, multi-agent AI system that separates distinct cognitive functions:

1.  **Simulation (`simulation.py`):** The world is modeled in PyBullet, featuring a Franka Emika Panda arm and dynamically placed objects. A virtual camera provides RGB and Depth data.

2.  **Perception (The "Eyes"):** A **Vision-Language Model (VLM)**, specifically `qwen3-vl:8b`, is used to analyze the RGB image from the camera. It is tasked with identifying the target object and returning its 2D bounding box. This approach was chosen after a comparative study proved it was more robust than specialized detectors in this synthetic domain.

3.  **Localization (The "Mapper"):** A robust **de-projection algorithm** converts the VLM's 2D pixel output into a 3D world coordinate `(x, y, z)`. This module uses the depth data from the camera to perform the 2D-to-3D transformation.

4.  **Planning (The "Brain"):** A pure **Language Model (`llama3:8b`)** acts as the high-level task planner. It receives the dynamically perceived 3D coordinate of the object and generates a safe, multi-step motion plan in a structured JSON format.

5.  **Action (The "Body"):** A deterministic **Robot Controller (`robot_controller.py`)** subscribes to the plan and executes each `move_to` command sequentially by calculating the inverse kinematics for the robot arm.

![Architecture Diagram]

---

## 🔬 Key Research Findings & Demonstrated Skills

This project serves as a practical demonstration of core concepts in AI and robotics:

*   **Comparative AI Analysis:** Rigorously tested and compared three different perception methodologies:
    1.  **VLM (Successful):** A general-purpose VLM (`qwen3-vl`) successfully bridged the Sim-to-Real domain gap, providing functional, if slightly imprecise, object localization.
    2.  **Specialized Detector (Failure):** A pre-trained YOLOv8m model consistently failed, confidently hallucinating incorrect objects (e.g., misidentifying the robot's base as a "traffic light"), proving the critical challenge of the domain gap for models trained only on real-world data.
    3.  **End-to-End API Model (Hybrid Success):** An advanced API-based model (Google's Gemini Pro) demonstrated remarkable speed and 3D reasoning capabilities, but highlighted the trade-off between end-to-end abstraction and mathematical precision.

*   **Advanced Prompt Engineering:** Developed and refined a series of sophisticated prompts to reliably control AI model output, employing techniques like:
    *   **Few-Shot Examples:** Providing in-context examples to guide model behavior.
    *   **Dynamic Contextual Bounding:** Explicitly stating environmental constraints (like image resolution) in the prompt to prevent hallucinations.

*   **Robust System Design:** Implemented fault-tolerant mechanisms, such as retry loops for the perception module, to handle the stochastic and sometimes unreliable nature of AI models.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.9+
*   Conda (recommended for environment management)
*   [Ollama](https://ollama.com/) installed and running for local model inference.

### Installation

1.  **Clone the repository:**

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate llm_robot_controller
    ```

3.  **Download the necessary local AI models:**
    ```bash
    ollama pull qwen3-vl
    ollama pull llama3:8b
    ```

4.  **(Optional) Set up Google Gemini API:**
    *   Create a `.env` file in the project root.
    *   Add your API key to the file: `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
    *   Install the required library: `pip install google.genai python-dotenv`

### Running the Main Application

To run the primary, successful VLM-based system:

**Start the Ollama server** in a separate terminal:
    ```bash
    ollama serve
    ```

### Running

*   **To run the Qwen + Llama flow:**
    ```bash
    python -m qwen_vlm_llama_llm
    ```

*   **To run the Gemini flow:**
    ```bash
    python -m gemini_end_to_end
    ```

---

## 🔮 Future Work

This project lays the foundation for several exciting research directions:

1.  **Closed-Loop Error Recovery:** Implement a feedback loop where the robot uses its camera *again* after a grasp attempt to verify success. If the object was not picked up, this failure state would be fed back to the planner LLM to generate a corrective action.
2.  **Domain-Specific Fine-Tuning:** Generate a synthetic dataset from the simulator to fine-tune the YOLO model, which would solve its domain gap issue and enable a high-speed, high-accuracy perception pipeline.
3.  **Physical Gripper Implementation:** Replace the simulated grasp (`time.sleep`) with a physical `pybullet.createConstraint` to allow the robot to complete the physical transfer of the object.