import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random
from collections import deque


class GestureRecognizer:
    """Recognizes hand gestures from MediaPipe landmarks."""
    
    def __init__(self):
        """Initialize gesture recognition."""
        self.current_gesture = "UNKNOWN"
        self.gesture_history = deque(maxlen=5)  # Smooth gesture changes
    
    def count_fingers(self, hand_landmarks):
        """
        Count extended fingers.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            int: Number of extended fingers (0-5)
        """
        fingers = []
        
        # Thumb (special case - check horizontal distance)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        if abs(thumb_tip.x - thumb_base.x) > 0.05:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (check if tip is above base joint)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_bases = [6, 10, 14, 18]
        
        for tip, base in zip(finger_tips, finger_bases):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def recognize_gesture(self, hand_landmarks):
        """
        Recognize specific hand gesture.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            str: Gesture name
        """
        finger_count = self.count_fingers(hand_landmarks)
        
        # Get key landmarks for advanced gesture detection
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Define gestures based on finger patterns
        if finger_count == 5:
            gesture = "OPEN_HAND"
        
        elif finger_count == 0:
            gesture = "FIST"
        
        elif finger_count == 1:
            # Check if it's pointing (index finger only)
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                gesture = "POINTING"
            else:
                gesture = "THUMBS_UP"
        
        elif finger_count == 2:
            # Peace sign vs Rock sign
            index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            pinky_extended = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            
            if index_extended and middle_extended and not pinky_extended:
                gesture = "PEACE"
            elif index_extended and pinky_extended:
                gesture = "ROCK"
            else:
                gesture = "TWO_FINGERS"
        
        elif finger_count == 3:
            gesture = "THREE_FINGERS"
        
        elif finger_count == 4:
            gesture = "FOUR_FINGERS"
        
        else:
            gesture = "UNKNOWN"
        
        # Add to history for smoothing
        self.gesture_history.append(gesture)
        
        # Return most common gesture in recent history
        if len(self.gesture_history) > 0:
            most_common = max(set(self.gesture_history), 
                            key=list(self.gesture_history).count)
            self.current_gesture = most_common
            return most_common
        
        return gesture



class Particle:
    """Individual particle with position, velocity, color, and lifetime."""
    
    def __init__(self, pos, color=None, velocity_scale=1.0):
        """Initialize a particle."""
        self.pos = np.array(pos, dtype=float)
        
        # Random velocity
        self.velocity = np.array([
            random.uniform(-0.02, 0.02),
            random.uniform(-0.02, 0.02),
            random.uniform(-0.02, 0.02)
        ]) * velocity_scale
        
        # Holographic colors
        if color is None:
            hue = random.choice([
                [0.3, 0.8, 1.0],  # Cyan
                [1.0, 0.3, 0.8],  # Magenta
                [1.0, 0.9, 0.3],  # Yellow
                [0.5, 0.3, 1.0],  # Purple
                [0.3, 1.0, 0.5],  # Green
            ])
            self.color = hue + [1.0]
        else:
            self.color = color
        
        self.life = 1.0
        self.decay = random.uniform(0.01, 0.03)
        self.size = random.uniform(2, 5)
    
    def update(self):
        """Update particle."""
        self.pos += self.velocity
        self.life -= self.decay
        self.velocity[1] -= 0.0005  # Gravity
        self.velocity += np.random.uniform(-0.001, 0.001, 3)  # Turbulence
        return self.life > 0
    
    def draw_2d(self, frame, fx, fy, fz_scale=200):
        """
        Draw particle on 2D frame.
        
        Args:
            frame: OpenCV frame to draw on
            fx, fy: Frame dimensions
            fz_scale: Scale for Z depth
        """
        # Convert 3D position to 2D screen coordinates
        x = int((self.pos[0] + 1) * fx / 2)
        y = int((1 - self.pos[1]) * fy / 2)
        
        # Size based on depth and life
        size = int(self.size * self.life * (1 + self.pos[2] / 2))
        size = max(1, min(size, 20))
        
        # Color with alpha
        color_bgr = (
            int(self.color[2] * 255 * self.life),  # B
            int(self.color[1] * 255 * self.life),  # G
            int(self.color[0] * 255 * self.life),  # R
        )
        
        # Draw glowing circle
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x, y), size, color_bgr, -1, cv2.LINE_AA)
            # Outer glow
            cv2.circle(frame, (x, y), size + 2, color_bgr, 1, cv2.LINE_AA)


class ParticleSystem:
    """Manages multiple particles."""
    
    def __init__(self, max_particles=800):
        """Initialize particle system."""
        self.particles = []
        self.max_particles = max_particles
    
    def emit(self, pos, count=5, velocity_scale=1.0, color=None):
        """Emit particles from position."""
        for _ in range(count):
            if len(self.particles) < self.max_particles:
                offset_pos = [
                    pos[0] + random.uniform(-0.05, 0.05),
                    pos[1] + random.uniform(-0.05, 0.05),
                    pos[2] + random.uniform(-0.05, 0.05)
                ]
                self.particles.append(Particle(offset_pos, color, velocity_scale))
    
    def explode(self, pos, count=50):
        """Create explosion effect."""
        for _ in range(count):
            if len(self.particles) < self.max_particles:
                velocity_scale = random.uniform(2, 4)
                self.particles.append(Particle(pos, velocity_scale=velocity_scale))
    
    def update(self):
        """Update all particles."""
        self.particles = [p for p in self.particles if p.update()]
    
    def draw_2d(self, frame):
        """Draw all particles on 2D frame."""
        fx, fy = frame.shape[1], frame.shape[0]
        for particle in self.particles:
            particle.draw_2d(frame, fx, fy)


def draw_shape_2d(frame, shape_type, pos, size=100):
    """
    Draw 3D wireframe shapes on 2D frame.
    
    Args:
        frame: OpenCV frame
        shape_type: Type of shape to draw
        pos: [x, y, z] position in normalized space
        size: Base size of shape
    """
    fx, fy = frame.shape[1], frame.shape[0]
    
    # Convert normalized position to screen coordinates
    cx = int((pos[0] + 1) * fx / 2)
    cy = int((1 - pos[1]) * fy / 2)
    
    # Scale size based on depth
    depth_scale = 1 + pos[2] / 2
    scaled_size = int(size * depth_scale)
    
    # Animated rotation
    time_val = pygame.time.get_ticks() / 1000.0
    angle = time_val * 30  # degrees per second
    
    # Holographic color (animated)
    r = int((math.sin(time_val * 0.5) + 1) * 127 + 64)
    g = int((math.sin(time_val * 0.7 + 2) + 1) * 127 + 64)
    b = int((math.sin(time_val * 0.9 + 4) + 1) * 127 + 64)
    color = (b, g, r)
    
    if shape_type == "SPHERE":
        draw_sphere_2d(frame, cx, cy, scaled_size, color, angle)
    
    elif shape_type == "CUBE":
        draw_cube_2d(frame, cx, cy, scaled_size, color, angle)
    
    elif shape_type == "PYRAMID":
        draw_pyramid_2d(frame, cx, cy, scaled_size, color, angle)
    
    elif shape_type == "TORUS":
        draw_torus_2d(frame, cx, cy, scaled_size, color, angle)
    
    elif shape_type == "HELIX":
        draw_helix_2d(frame, cx, cy, scaled_size, color, angle)
    
    elif shape_type == "SPIRAL":
        draw_spiral_2d(frame, cx, cy, scaled_size, color, angle)


def draw_sphere_2d(frame, cx, cy, size, color, angle):
    """Draw wireframe sphere."""
    segments = 16
    
    # Draw latitude circles
    for i in range(segments // 2):
        lat = (i / (segments // 2) - 0.5) * math.pi
        r = int(size * math.cos(lat))
        y_offset = int(size * math.sin(lat))
        cv2.circle(frame, (cx, cy + y_offset), r, color, 2, cv2.LINE_AA)
    
    # Draw longitude circles
    for i in range(4):
        angle_i = i * math.pi / 4
        points = []
        for j in range(segments):
            t = j / segments * 2 * math.pi
            x = int(cx + size * math.sin(t) * math.cos(angle_i))
            y = int(cy + size * math.cos(t))
            points.append((x, y))
        
        for j in range(len(points) - 1):
            cv2.line(frame, points[j], points[j + 1], color, 2, cv2.LINE_AA)


def draw_cube_2d(frame, cx, cy, size, color, angle):
    """Draw rotating wireframe cube."""
    # Cube vertices
    s = size / 2
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])
    
    # Rotation matrices
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
    
    # Apply rotation
    rotated = vertices @ rot_y.T @ rot_x.T
    
    # Project to 2D
    points = []
    for v in rotated:
        # Simple perspective projection
        scale = 200 / (200 + v[2])
        x = int(cx + v[0] * scale)
        y = int(cy + v[1] * scale)
        points.append((x, y))
    
    # Draw cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    for edge in edges:
        cv2.line(frame, points[edge[0]], points[edge[1]], color, 2, cv2.LINE_AA)


def draw_pyramid_2d(frame, cx, cy, size, color, angle):
    """Draw rotating pyramid."""
    s = size
    vertices = np.array([
        [0, -s, 0],      # Apex
        [-s, s, -s],     # Base corners
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


def draw_torus_2d(frame, cx, cy, size, color, angle):
    """Draw rotating torus."""
    major_r = size * 0.6
    minor_r = size * 0.3
    segments = 32
    
    rad = math.radians(angle)
    
    for i in range(segments):
        theta = i / segments * 2 * math.pi
        points = []
        
        for j in range(segments):
            phi = j / segments * 2 * math.pi
            
            # Torus parametric equations
            x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
            y = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
            z = minor_r * math.sin(phi)
            
            # Rotate
            x_rot = x * math.cos(rad) - z * math.sin(rad)
            z_rot = x * math.sin(rad) + z * math.cos(rad)
            
            # Project
            scale = 200 / (200 + z_rot)
            px = int(cx + x_rot * scale)
            py = int(cy + y * scale)
            points.append((px, py))
        
        # Draw ring
        for j in range(len(points) - 1):
            cv2.line(frame, points[j], points[j + 1], color, 1, cv2.LINE_AA)


def draw_helix_2d(frame, cx, cy, size, color, angle):
    """Draw DNA-like double helix."""
    height = size * 2
    radius = size * 0.5
    segments = 50
    
    rad = math.radians(angle)
    
    points1 = []
    points2 = []
    
    for i in range(segments):
        t = i / segments
        theta = t * 4 * math.pi + rad
        
        # First helix strand
        x1 = radius * math.cos(theta)
        y1 = height * (t - 0.5)
        z1 = radius * math.sin(theta)
        
        # Second helix strand (opposite phase)
        x2 = radius * math.cos(theta + math.pi)
        y2 = y1
        z2 = radius * math.sin(theta + math.pi)
        
        # Project to 2D
        scale1 = 200 / (200 + z1)
        scale2 = 200 / (200 + z2)
        
        p1 = (int(cx + x1 * scale1), int(cy + y1 * scale1))
        p2 = (int(cx + x2 * scale2), int(cy + y2 * scale2))
        
        points1.append(p1)
        points2.append(p2)
        
        # Draw connecting bars every few segments
        if i % 5 == 0:
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
    
    # Draw helix strands
    for i in range(len(points1) - 1):
        cv2.line(frame, points1[i], points1[i + 1], color, 2, cv2.LINE_AA)
        cv2.line(frame, points2[i], points2[i + 1], color, 2, cv2.LINE_AA)


def draw_spiral_2d(frame, cx, cy, size, color, angle):
    """Draw 3D spiral."""
    segments = 100
    rad = math.radians(angle)
    
    points = []
    for i in range(segments):
        t = i / segments
        theta = t * 6 * math.pi + rad
        r = size * t
        
        x = r * math.cos(theta)
        y = size * (t - 0.5) * 2
        z = r * math.sin(theta)
        
        # Rotate
        x_rot = x * math.cos(rad * 0.5) - z * math.sin(rad * 0.5)
        z_rot = x * math.sin(rad * 0.5) + z * math.cos(rad * 0.5)
        
        # Project
        scale = 200 / (200 + z_rot)
        px = int(cx + x_rot * scale)
        py = int(cy + y * scale)
        points.append((px, py))
    
    # Draw spiral
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)




class HandTracker:
    """MediaPipe hand tracking with 3D coordinate estimation."""
    
    def __init__(self):
        """Initialize MediaPipe hands."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.position_history = []
        self.max_history = 10
    
    def process_frame(self, frame):
        """Process frame and detect hand."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on frame."""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
    
    def get_3d_position(self, hand_landmarks):
        """Convert 2D landmarks to 3D position."""
        # Palm center
        wrist = hand_landmarks.landmark[0]
        index_mcp = hand_landmarks.landmark[5]
        pinky_mcp = hand_landmarks.landmark[17]
        
        palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
        palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3
        
        # Depth estimation
        middle_mcp = hand_landmarks.landmark[9]
        hand_size = math.sqrt(
            (wrist.x - middle_mcp.x) ** 2 + 
            (wrist.y - middle_mcp.y) ** 2
        )
        
        x = (palm_x - 0.5) * 2
        y = -(palm_y - 0.5) * 2
        z = (hand_size - 0.15) * 3
        
        position = [x, y, z]
        
        # Smoothing
        self.position_history.append(position)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        avg_pos = np.mean(self.position_history, axis=0)
        return avg_pos.tolist()
    
    def close(self):
        """Release resources."""
        self.hands.close()




class GestureParticleSystem:
    """Main application with gesture-controlled shapes."""
    
    def __init__(self):
        """Initialize application."""
        # Initialize Pygame (for timing only)
        pygame.init()
        self.clock = pygame.time.Clock()
        
        # Components
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.particle_system = ParticleSystem(max_particles=1000)
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # State
        self.hand_position = [0, 0, 0]
        self.current_gesture = "NONE"
        self.trail_positions = deque(maxlen=50)
        self.running = True
        self.show_help = True
        
        # Gesture to shape mapping
        self.gesture_shapes = {
            "OPEN_HAND": "SPHERE",
            "PEACE": "CUBE",
            "POINTING": "SPIRAL",
            "FIST": "EXPLOSION",
            "ROCK": "TORUS",
            "THUMBS_UP": "PYRAMID",
            "FOUR_FINGERS": "HELIX",
            "THREE_FINGERS": "SPHERE",
        }
        
        # Explosion cooldown
        self.last_explosion = 0
        self.explosion_cooldown = 1.0  # seconds
    
    def draw_help_text(self, frame):
        """Draw help overlay."""
        if not self.show_help:
            return
        
        help_text = [
            "GESTURES:",
            "Open Hand (5) - Sphere",
            "Peace (2) - Cube",
            "Pointing (1) - Spiral",
            "Fist (0) - Explosion",
            "Rock Sign - Torus",
            "Thumbs Up - Pyramid",
            "4 Fingers - DNA Helix",
            "",
            "Press H - Toggle Help",
            "Press Q - Quit"
        ]
        
        y = 30
        for i, text in enumerate(help_text):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
            y += 25
    
    def draw_gesture_info(self, frame):
        """Draw current gesture info."""
        gesture_text = f"Gesture: {self.current_gesture}"
        shape = self.gesture_shapes.get(self.current_gesture, "NONE")
        shape_text = f"Shape: {shape}"
        
        cv2.putText(frame, gesture_text, (frame.shape[1] - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, shape_text, (frame.shape[1] - 300, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    
    def update(self):
        """Update tracking and particles."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)  # Mirror for intuitive control
        
        # Process hand tracking
        results = self.hand_tracker.process_frame(frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.hand_tracker.draw_landmarks(frame, hand_landmarks)
            
            # Get 3D position
            self.hand_position = self.hand_tracker.get_3d_position(hand_landmarks)
            
            # Recognize gesture
            self.current_gesture = self.gesture_recognizer.recognize_gesture(hand_landmarks)
            
            # Add to trail
            self.trail_positions.append(self.hand_position.copy())
            
            # Handle gesture-specific effects
            shape_type = self.gesture_shapes.get(self.current_gesture, "SPHERE")
            
            if shape_type == "EXPLOSION":
                # Explosion on fist (with cooldown)
                current_time = pygame.time.get_ticks() / 1000.0
                if current_time - self.last_explosion > self.explosion_cooldown:
                    self.particle_system.explode(self.hand_position, count=100)
                    self.last_explosion = current_time
            else:
                # Regular particle emission
                particle_count = 15 if shape_type == "SPIRAL" else 8
                self.particle_system.emit(self.hand_position, count=particle_count)
            
            # Draw shape
            if shape_type != "EXPLOSION":
                draw_shape_2d(frame, shape_type, self.hand_position, size=120)
        
        else:
            self.current_gesture = "NONE"
        
        # Update particles
        self.particle_system.update()
        
        # Draw particles on frame
        self.particle_system.draw_2d(frame)
        
        # Draw trail
        if len(self.trail_positions) > 1:
            fx, fy = frame.shape[1], frame.shape[0]
            for i in range(len(self.trail_positions) - 1):
                p1 = self.trail_positions[i]
                p2 = self.trail_positions[i + 1]
                
                x1 = int((p1[0] + 1) * fx / 2)
                y1 = int((1 - p1[1]) * fy / 2)
                x2 = int((p2[0] + 1) * fx / 2)
                y2 = int((1 - p2[1]) * fy / 2)
                
                alpha = i / len(self.trail_positions)
                color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        
        self.draw_help_text(frame)
        self.draw_gesture_info(frame)
        
        fps = self.clock.get_fps()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main loop."""
        print("=" * 70)
        print("GESTURE-CONTROLLED 3D PARTICLE SYSTEM")
        print("=" * 70)
        print("\nGestures:")
        print("  ✋ Open Hand (5 fingers)  → Holographic Sphere")
        print("  ✌️  Peace Sign (2 fingers) → Rotating Cube")
        print("  👆 Pointing (1 finger)    → Spiral Trail")
        print("  ✊ Fist (0 fingers)       → Particle Explosion")
        print("  🤘 Rock Sign              → Torus Ring")
        print("  👍 Thumbs Up              → Pyramid")
        print("  🖖 4 Fingers              → DNA Helix")
        print("\nControls:")
        print("  H - Toggle help overlay")
        print("  Q or ESC - Quit")
        print("=" * 70)
        
        while self.running:
            frame = self.update()
            
            if frame is not None:
                cv2.imshow('Gesture-Controlled Particle System', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                self.running = False
            elif key == ord('h'):  # Toggle help
                self.show_help = not self.show_help
            
            self.clock.tick(60)  # 60 FPS
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.hand_tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    app = GestureParticleSystem()
    app.run()
