import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
], dtype=np.float32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    cap = cv2.VideoCapture(0)

    model = YOLO("yolov8l.pt")

    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1
    )

    # Fetch frame dimensions to calculate the polygon zone correctly
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return

    frame_height, frame_width = frame.shape[:2]
    zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        # Extract detections manually since from_yolov8 is not available
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
        labels = [
            f"{model.names[int(class_id)]} {confidence:0.2f}"
            for confidence, class_id in zip(confidences, class_ids)
        ]
        frame = bounding_box_annotator.annotate(
            scene=frame,
            detections=detections
        )
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Attempt to display the frame, handle errors if any
        try:
            cv2.imshow("yolov8", frame)
        except cv2.error as e:
            print(f"Error displaying frame: {e}")
            # Optional: Save frame to disk instead
            cv2.imwrite("output_frame.jpg", frame)
            break

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
