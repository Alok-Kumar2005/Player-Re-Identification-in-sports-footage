from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
from utils import Sort
from sklearn.metrics.pairwise import cosine_similarity
import math

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.3)
        
        # ID consistency features
        self.player_features = {}
        self.player_positions = {}
        self.inactive_players = {}
        self.id_mapping = {}  # Maps SORT IDs to consistent IDs
        self.next_consistent_id = 1
        self.max_inactive_frames = 30
        
    def extract_features(self, frame, bbox):
        """Extract simple color features from player region"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure valid bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(96)
            
            player_region = frame[y1:y2, x1:x2]
            
            # Use top half for jersey color
            mid_y = player_region.shape[0] // 2
            jersey_region = player_region[:mid_y, :]
            
            if jersey_region.size == 0:
                return np.zeros(96)
            
            # Resize to standard size
            jersey_region = cv2.resize(jersey_region, (32, 32))
            
            # Extract color histogram
            hist_b = cv2.calcHist([jersey_region], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([jersey_region], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([jersey_region], [2], None, [32], [0, 256])
            
            features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(96)
    
    def calculate_similarity(self, feat1, feat2, pos1, pos2):
        """Calculate similarity between two players based on features and position"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Feature similarity
        feature_sim = cosine_similarity([feat1], [feat2])[0][0]
        
        # Position similarity
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        position_sim = max(0, 1 - distance / 200)  # Normalize by max expected movement
        
        # Combined similarity
        return 0.7 * feature_sim + 0.3 * position_sim
    
    def find_best_match(self, frame, bbox, frame_num):
        """Find the best matching inactive player for reactivation"""
        if not self.inactive_players:
            return None
        
        current_features = self.extract_features(frame, bbox)
        current_pos = get_center_of_bbox(bbox)
        
        best_match = None
        best_similarity = 0.4  # Minimum threshold
        
        for player_id, info in self.inactive_players.items():
            if frame_num - info['lost_frame'] > self.max_inactive_frames:
                continue
            
            similarity = self.calculate_similarity(
                current_features, 
                info['features'], 
                current_pos, 
                info['position']
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = player_id
        
        return best_match
    
    def update_id_mapping(self, sort_tracks, frame, frame_num):
        """Update the mapping between SORT IDs and consistent IDs"""
        current_sort_ids = set()
        
        for track in sort_tracks:
            sort_id = int(track[4])
            current_sort_ids.add(sort_id)
            bbox = track[:4].tolist()
            
            if sort_id in self.id_mapping:
                # Existing track - update info
                consistent_id = self.id_mapping[sort_id]
                self.player_features[consistent_id] = self.extract_features(frame, bbox)
                self.player_positions[consistent_id] = get_center_of_bbox(bbox)
                
                # Remove from inactive if present
                if consistent_id in self.inactive_players:
                    del self.inactive_players[consistent_id]
                    
            else:
                # New SORT ID - check if it matches an inactive player
                best_match = self.find_best_match(frame, bbox, frame_num)
                
                if best_match:
                    # Reactivate existing player
                    self.id_mapping[sort_id] = best_match
                    self.player_features[best_match] = self.extract_features(frame, bbox)
                    self.player_positions[best_match] = get_center_of_bbox(bbox)
                    del self.inactive_players[best_match]
                    print(f"Reactivated player {best_match} with new SORT ID {sort_id}")
                else:
                    # Create new consistent ID
                    new_id = self.next_consistent_id
                    self.next_consistent_id += 1
                    self.id_mapping[sort_id] = new_id
                    self.player_features[new_id] = self.extract_features(frame, bbox)
                    self.player_positions[new_id] = get_center_of_bbox(bbox)
                    print(f"Created new player {new_id} with SORT ID {sort_id}")
        
        # Handle lost tracks
        lost_sort_ids = set(self.id_mapping.keys()) - current_sort_ids
        for lost_sort_id in lost_sort_ids:
            consistent_id = self.id_mapping[lost_sort_id]
            
            # Move to inactive
            self.inactive_players[consistent_id] = {
                'features': self.player_features.get(consistent_id),
                'position': self.player_positions.get(consistent_id),
                'lost_frame': frame_num
            }
            
            # Remove from active mapping
            del self.id_mapping[lost_sort_id]
            print(f"Lost player {consistent_id} (SORT ID {lost_sort_id})")
        
        # Clean up old inactive players
        to_remove = []
        for player_id, info in self.inactive_players.items():
            if frame_num - info['lost_frame'] > self.max_inactive_frames:
                to_remove.append(player_id)
        
        for player_id in to_remove:
            del self.inactive_players[player_id]
            if player_id in self.player_features:
                del self.player_features[player_id]
            if player_id in self.player_positions:
                del self.player_positions[player_id]

    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.8)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Prepare detections
            player_detections = []
            referee_detections = []
            ball_detections = []
            
            for i, (bbox, confidence, class_id) in enumerate(zip(
                detection_supervision.xyxy, 
                detection_supervision.confidence, 
                detection_supervision.class_id
            )):
                detection_array = np.array([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
                
                if class_id == cls_names_inv['player']:
                    player_detections.append(detection_array)
                elif class_id == cls_names_inv['referee']:
                    referee_detections.append(detection_array)
                elif class_id == cls_names_inv['ball']:
                    ball_detections.append(detection_array)

            # Track players using SORT
            if len(player_detections) > 0:
                sort_tracks = self.tracker.update(np.array(player_detections))
            else:
                sort_tracks = np.empty((0, 5))

            # Update ID mapping
            self.update_id_mapping(sort_tracks, frames[frame_num], frame_num)

            # Initialize frame dictionaries
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process player tracks with consistent IDs
            for track in sort_tracks:
                bbox = track[:4].tolist()
                sort_id = int(track[4])
                
                if sort_id in self.id_mapping:
                    consistent_id = self.id_mapping[sort_id]
                    tracks["players"][frame_num][consistent_id] = {"bbox": bbox}

            # Process referees and ball (unchanged)
            for detection_array in referee_detections:
                bbox = detection_array[:4].tolist()
                tracks["referees"][frame_num][len(tracks["referees"][frame_num])] = {"bbox": bbox}
            
            for detection_array in ball_detections:
                bbox = detection_array[:4].tolist()
                tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames