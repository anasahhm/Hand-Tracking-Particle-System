import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque


class GestureRecognizer:
    """Recognizes hand gestures."""
    
    def __init__(self):
        """Initialize gesture recognition."""
        self.current_gesture = "NONE"
        self.gesture_history = deque(maxlen=5)
    
    def count_fingers(self, hand_landmarks):
        """Count extended fingers."""
        fingers = []
        
        # Thumb
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        if abs(thumb_tip.x - thumb_base.x) > 0.05:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_bases = [6, 10, 14, 18]
        
        for tip, base in zip(finger_tips, finger_bases):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y - 0.02:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def detect_pinch(self, hand_landmarks):
        """Detect pinch gesture and return distance."""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        is_pinching = distance < 0.08
        return is_pinching, distance
    
    def recognize_gesture(self, hand_landmarks):
        """Recognize gesture."""
        finger_count = self.count_fingers(hand_landmarks)
        is_pinching, pinch_dist = self.detect_pinch(hand_landmarks)
        
        # Priority: Pinch > Finger count
        if is_pinching:
            gesture = "PINCH"
        elif finger_count == 5:
            gesture = "OPEN_HAND"
        elif finger_count == 2:
            # Check for peace sign
            index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            if index_extended and middle_extended:
                gesture = "PEACE"
            else:
                gesture = "OPEN_HAND"
        elif finger_count == 1:
            # Pointing
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                gesture = "POINTING"
            else:
                gesture = "OPEN_HAND"
        else:
            gesture = "NONE"
        
        # Smooth with history
        self.gesture_history.append(gesture)
        most_common = max(set(self.gesture_history), key=list(self.gesture_history).count)
        self.current_gesture = most_common
        
        return most_common, pinch_dist


def draw_sphere(frame, cx, cy, size, color):
    """Draw clean wireframe sphere."""
    segments = 12
    
    # Draw latitude circles
    for i in range(segments // 2):
        lat = (i / (segments // 2) - 0.5) * math.pi
        r = int(size * abs(math.cos(lat)))
        y_offset = int(size * math.sin(lat))
        if r > 5:
            cv2.circle(frame, (cx, cy + y_offset), r, color, 2, cv2.LINE_AA)
    
    # Draw longitude circles
    for i in range(4):
        angle = i * math.pi / 4
        points = []
        for j in range(segments):
            t = j / segments * 2 * math.pi
            x = int(cx + size * math.sin(t) * math.cos(angle))
            y = int(cy + size * math.cos(t))
            points.append((x, y))
        
        for j in range(len(points) - 1):
            cv2.line(frame, points[j], points[j + 1], color, 2, cv2.LINE_AA)


def draw_cube(frame, cx, cy, size, color, angle):
    """Draw clean rotating cube."""
    s = size / 2
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])
    
    # Rotation
    rad = math.radians(angle)
    rot_y = np.array([
        [math.cos(rad), 0, math.sin(rad)],
        [0, 1, 0],
        [-math.sin(rad), 0, math.cos(rad)]
    ])
    
    rot_x = np.array([
        [1, 0, 0],
        [0, math.cos(rad * 0.5), -math.sin(rad * 0.5)],
        [0, math.sin(rad * 0.5), math.cos(rad * 0.5)]
    ])
    
    rotated = vertices @ rot_y.T @ rot_x.T
    
    # Project to 2D
    points = []
    for v in rotated:
        scale = 200 / (200 + v[2])
        x = int(cx + v[0] * scale)
        y = int(cy + v[1] * scale)
        points.append((x, y))
    
    # Draw edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for edge in edges:
        cv2.line(frame, points[edge[0]], points[edge[1]], color, 2, cv2.LINE_AA)


def draw_pyramid(frame, cx, cy, size, color, angle):
    """Draw clean rotating pyramid."""
    s = size
    vertices = np.array([
        [0, -s, 0],
        [-s, s, -s],
        [s, s, -s],
        [s, s, s],
        [-s, s, s]
    ])
    
    # Rotation
    rad = math.radians(angle)
    rot_y = np.array([
        [math.cos(rad), 0, math.sin(rad)],
        [0, 1, 0],
        [-math.sin(rad), 0, math.cos(rad)]
    ])
    
    rotated = vertices @ rot_y.T
    
    # Project to 2D
    points = []
    for v in rotated:
        scale = 200 / (200 + v[2])
        x = int(cx + v[0] * scale)
        y = int(cy + v[1] * scale)
        points.append((x, y))
    
    # Draw edges
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    for edge in edges:
        cv2.line(frame, points[edge[0]], points[edge[1]], color, 2, cv2.LINE_AA)




class HandTracker:
    """MediaPipe hand tracking."""
    
    def __init__(self):
        """Initialize MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.position_history = deque(maxlen=5)
    
    def process_frame(self, frame):
        """Process frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks."""
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
    
    def get_position(self, hand_landmarks):
        """Get hand position."""
        wrist = hand_landmarks.landmark[0]
        middle = hand_landmarks.landmark[9]
        
        palm_x = (wrist.x + middle.x) / 2
        palm_y = (wrist.y + middle.y) / 2
        
        x = (palm_x - 0.5) * 2
        y = -(palm_y - 0.5) * 2 - 0.3
        
        position = [x, y]
        self.position_history.append(position)
        
        return np.mean(self.position_history, axis=0).tolist()
    
    def close(self):
        """Release resources."""
        self.hands.close()




class CleanGestureApp:
    """Clean gesture-controlled shapes application."""
    
    def __init__(self):
        """Initialize application."""
        # Components
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # State
        self.hand_position = [0, 0]
        self.current_gesture = "NONE"
        self.current_shape = "SPHERE"
        self.shape_size = 100
        self.angle = 0
        self.running = True
        self.show_help = True
        
        # Gesture to shape mapping
        self.gesture_shapes = {
            "OPEN_HAND": "SPHERE",
            "PEACE": "CUBE",
            "POINTING": "PYRAMID",
        }
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = cv2.getTickCount()
    
    def draw_ui(self, frame):
        """Draw UI elements."""
        h, w = frame.shape[:2]
        
        # Title
        cv2.rectangle(frame, (10, 10), (280, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, 60), (0, 255, 255), 2)
        cv2.putText(frame, "GESTURE SHAPES", (25, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Current state
        state_y = 80
        cv2.putText(frame, f"Shape: {self.current_shape}", (10, state_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Size: {int(self.shape_size)}", (10, state_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, state_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2, cv2.LINE_AA)
        
        # Help text
        if self.show_help:
            help_lines = [
                "GESTURES:",
                "Open Hand (5) - Sphere",
                "Peace Sign (2) - Cube",
                "Pointing (1) - Pyramid",
                "Pinch - Resize shape",
                "",
                "H - Toggle Help",
                "Q - Quit"
            ]
            
            y = h - 220
            for i, line in enumerate(help_lines):
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                cv2.putText(frame, line, (10, y + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # FPS
        if len(self.fps_history) > 0:
            fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    def update(self):
        """Main update loop."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process hand tracking
        results = self.hand_tracker.process_frame(frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.hand_tracker.draw_landmarks(frame, hand_landmarks)
            
            # Get hand position
            self.hand_position = self.hand_tracker.get_position(hand_landmarks)
            
            # Recognize gesture
            gesture, pinch_dist = self.gesture_recognizer.recognize_gesture(hand_landmarks)
            self.current_gesture = gesture
            
            # Update shape based on gesture
            if gesture in self.gesture_shapes:
                self.current_shape = self.gesture_shapes[gesture]
            
            # Resize with pinch
            if gesture == "PINCH":
                # Map pinch distance to size (0.02-0.08 -> 200-50)
                self.shape_size = int(200 - (pinch_dist - 0.02) * 2500)
                self.shape_size = max(30, min(250, self.shape_size))
            
            # Calculate screen position
            cx = int((self.hand_position[0] + 1) * w / 2)
            cy = int((1 - self.hand_position[1]) * h / 2)
            
            # Update rotation angle
            self.angle += 1
            if self.angle >= 360:
                self.angle = 0
            
            # Get animated color
            t = cv2.getTickCount() / cv2.getTickFrequency()
            r = int((math.sin(t * 0.5) + 1) * 127 + 64)
            g = int((math.sin(t * 0.7 + 2) + 1) * 127 + 64)
            b = int((math.sin(t * 0.9 + 4) + 1) * 127 + 64)
            color = (b, g, r)
            
            # Draw shape
            if self.current_shape == "SPHERE":
                draw_sphere(frame, cx, cy, int(self.shape_size), color)
            elif self.current_shape == "CUBE":
                draw_cube(frame, cx, cy, int(self.shape_size), color, self.angle)
            elif self.current_shape == "PYRAMID":
                draw_pyramid(frame, cx, cy, int(self.shape_size), color, self.angle)
        
        # Draw UI
        self.draw_ui(frame)
        
        # Calculate FPS
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        return frame
    
    def run(self):
        """Main loop."""
        print("=" * 70)
        print("CLEAN GESTURE-CONTROLLED SHAPES")
        print("=" * 70)
        print("\nGestures:")
        print("  ✋ Open Hand (5 fingers) → Sphere")
        print("  ✌️  Peace Sign (2 fingers) → Cube")
        print("  👆 Pointing (1 finger)   → Pyramid")
        print("  🤏 Pinch                 → Resize shape")
        print("\nControls:")
        print("  H - Toggle help")
        print("  Q - Quit")
        print("=" * 70)
        
        while self.running:
            frame = self.update()
            
            if frame is not None:
                cv2.imshow('Clean Gesture Shapes', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
            elif key == ord('h'):
                self.show_help = not self.show_help
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.hand_tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = CleanGestureApp()
    app.run()
