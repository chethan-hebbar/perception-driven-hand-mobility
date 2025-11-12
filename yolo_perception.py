# src/yolo_perception.py
from ultralytics import YOLO

class YOLOPerception:
    """
    A class to handle object detection using a YOLOv8 model.
    """
    def __init__(self, model_name="yolov8m.pt"):
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        print(f"YOLOPerception module initialized with model: {model_name}")

    def find_target_object(self, image, target_class_name, conf_threshold=0.25):
        results = self.model(image, verbose=False)
        best_detection = None
        highest_confidence = conf_threshold

        for r in results:
            for box in r.boxes:
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.class_names[class_id]

                if class_name == target_class_name and confidence > highest_confidence:
                    highest_confidence = confidence
                    bbox = box.xyxy[0].tolist()
                    best_detection = {
                        "class_name": class_name,
                        "bbox": [int(coord) for coord in bbox],
                        "confidence": confidence
                    }
        return best_detection

    def find_all_objects(self, image, conf_threshold=0.25):
        """
        Finds ALL object detections of ANY class above a confidence threshold.
        """
        results = self.model(image, verbose=False)
        all_detections = []

        for r in results:
            for box in r.boxes:
                confidence = box.conf[0].item()
                if confidence > conf_threshold:
                    class_id = int(box.cls[0].item())
                    class_name = self.class_names[class_id]
                    bbox = box.xyxy[0].tolist()
                    all_detections.append({
                        "class_name": class_name,
                        "bbox": [int(coord) for coord in bbox],
                        "confidence": confidence
                    })
        return all_detections


if __name__ == '__main__':
    from src.simulation import Simulation
    import pybullet as p
    import cv2

    print("--- Testing YOLOPerception Class (Find All Objects Method) ---")
    sim = Simulation()
    perception = YOLOPerception()

    obj_pos = [0.5, 0.0, 0.05]
    obj_orn = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = sim.load_object("cube.urdf", obj_pos, orientation=obj_orn, global_scaling=0.1)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    print("Created a large red cube.")
    sim.run(1)

    print("\nCapturing image from simulation...")
    image, _ = sim.get_camera_images()

    print("Running GENERIC object detection to find ALL objects...")
    detections = perception.find_all_objects(image, conf_threshold=0.25)

    if detections:
        print(f"\n--- Found {len(detections)} object(s) with confidence > 0.25 ---")
        image_to_draw = image.copy()
        
        for i, detection in enumerate(detections):
            print(f"  Detection {i+1}:")
            print(f"    - YOLO labeled it as a: '{detection['class_name']}'")
            print(f"    - Confidence: {detection['confidence']:.2f}, BBox: {detection['bbox']}")
            
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_to_draw, f"{detection['class_name']} {detection['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print("\nDisplaying image with ALL YOLO detections. Press any key to close.")
        cv2.imshow("YOLO Detections", cv2.cvtColor(image_to_draw, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n--- No object detected with confidence > 0.25 ---")
        print("This confirms the Sim-to-Real domain gap is the core issue.")