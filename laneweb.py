import threading
import time

import cv2
import numpy as np

from MotorModule import Motor


class DetectionSystem:
    def __init__(self):
        # Initialize motor
        self.motor = Motor(2, 3, 4, 12, 22, 27)

        # Shared variables with thread locks
        self.frame = None
        self.curve = 0
        self.obstacle_detected = False
        self.running = True
        self.frame_lock = threading.Lock()

        # Lane detection parameters
        self.curve_list = []
        self.avg_val = 10

        # Object detection parameters
        self.min_area = 300
        self.danger_threshold = 50

    def create_dashboard(self, main_frame, mask_frame, edges_frame):
        """Combine all visualizations into one dashboard view"""
        height, width = main_frame.shape[:2]

        # Create a larger canvas for dashboard (2x original width to fit visualizations)
        dashboard = np.zeros((height, width * 2, 3), dtype=np.uint8)

        # Convert single channel images to 3-channel
        if len(mask_frame.shape) == 2:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
        if len(edges_frame.shape) == 2:
            edges_frame = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2BGR)

        # Resize auxiliary frames to fit dashboard
        small_height = height // 2
        small_width = width // 2
        mask_frame = cv2.resize(mask_frame, (small_width, small_height))
        edges_frame = cv2.resize(edges_frame, (small_width, small_height))

        # Place frames in dashboard
        # Main frame on left
        dashboard[0:height, 0:width] = main_frame

        # Mask and edges on right
        dashboard[0:small_height, width : width + small_width] = mask_frame
        dashboard[small_height:height, width : width + small_width] = edges_frame

        # Add labels
        cv2.putText(
            dashboard,
            "Main View",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            dashboard,
            "Object Mask",
            (width + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            dashboard,
            "Lane Detection",
            (width + 10, small_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        return dashboard

    def detect_objects(self, frame):
        height, width = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color detection (two ranges)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacle_detected = False
        closest_y = 0

        # Draw danger zone line
        danger_y = height - self.danger_threshold
        cv2.line(frame, (0, danger_y), (width, danger_y), (0, 255, 255), 2)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bottom_y = y + h

                if bottom_y > closest_y:
                    closest_y = bottom_y

                if bottom_y > height - self.danger_threshold:
                    color = (0, 0, 255)  # Red
                    obstacle_detected = True
                else:
                    color = (0, 255, 0)  # Green

                # Draw bounding box and information
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"Dist: {height - bottom_y}px",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Area: {area:.0f}",
                    (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Add status overlay
        status = "OBSTACLE DETECTED" if obstacle_detected else "NO OBSTACLE"
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if obstacle_detected else (0, 255, 0),
            2,
        )

        return frame, mask, obstacle_detected

    def process_lane_detection(self, frame):
        height, width = frame.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Define ROI
        roi_vertices = np.array(
            [
                [
                    (50, height),
                    (width - 50, height),
                    (width // 2 + 50, height // 2),
                    (width // 2 - 50, height // 2),
                ]
            ],
            dtype=np.int32,
        )

        # Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines
        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=50
        )

        curve = 0
        line_image = np.zeros_like(frame)

        if lines is not None:
            left_lines = []
            right_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0

                if slope < 0:  # Left lane
                    left_lines.append(line)
                elif slope > 0:  # Right lane
                    right_lines.append(line)

                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if left_lines and right_lines:
                left_avg = np.mean([l[0] for l in left_lines], axis=0)
                right_avg = np.mean([l[0] for l in right_lines], axis=0)
                curve = (right_avg[0] - left_avg[0]) / width

        # Smooth curve
        self.curve_list.append(curve)
        if len(self.curve_list) > self.avg_val:
            self.curve_list.pop(0)
        smooth_curve = np.mean(self.curve_list)

        # Combine images
        combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # Add curve information
        cv2.putText(
            combined,
            f"Curve: {smooth_curve:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        return smooth_curve, combined, masked_edges

    def frame_processing_thread(self):
        # Try different camera indices
        camera_indices = [1, 0, 2]
        cap = None

        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Successfully opened camera {idx}")
                break

        if not cap.isOpened():
            print("Error: Could not open any camera")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            frame = cv2.resize(frame, (480, 240))

            with self.frame_lock:
                # Process lanes
                curve, lane_frame, edges = self.process_lane_detection(frame)

                # Detect objects
                final_frame, mask, obstacle_detected = self.detect_objects(lane_frame)

                self.curve = curve
                self.obstacle_detected = obstacle_detected

                # Create and show dashboard
                dashboard = self.create_dashboard(final_frame, mask, edges)
                cv2.imshow("Detection System Dashboard", dashboard)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

        cap.release()
        cv2.destroyAllWindows()

    def motor_control_thread(self):
        while self.running:
            with self.frame_lock:
                curve = self.curve
                obstacle = self.obstacle_detected

            if obstacle:
                print("Obstacle detected - Stopping")
                self.motor.stop()
            else:
                curve_speed = np.clip(curve, -0.3, 0.3)
                self.motor.move(0.2, -curve_speed, 0.05)

            time.sleep(0.1)

    def run(self):
        processing_thread = threading.Thread(target=self.frame_processing_thread)
        motor_thread = threading.Thread(target=self.motor_control_thread)

        processing_thread.start()
        motor_thread.start()

        processing_thread.join()
        motor_thread.join()


if __name__ == "__main__":
    system = DetectionSystem()
    system.run()
