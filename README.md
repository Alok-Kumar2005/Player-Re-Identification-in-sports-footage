# Player-Re-Identification-in-sports-footage

## Installlation and setup
```
git clone https://github.com/Alok-Kumar2005/Player-Re-Identification-in-sports-footage.git
cd Player-Re-Identification-in-sports-footage
```

## Create Environment and activate it
```
python -m venv venv
venv/Scripts/activate
```

## Download the requirements.txt
```
pip install -r requirements2.txt
```

## Run 
```
python main.py
```



## Project Structure
├── main.py                 # Main execution script
├── trackers/
│   ├── __init__.py
│   └── tracker.py          # YOLO detection and SORT tracking implementation
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py    # Team color assignment using K-means
├── utils/
│   ├── __init__.py
│   ├── video_utils.py      # Video reading and saving utilities
│   └── bbox_utils.py       # Bounding box utility functions
|   └── sort.py                 # SORT tracking algorithm implementation
├── model/
│   └── best.pt             # Trained YOLO model weights
├── video/
│   └── 15sec_input_720p.mp4 # Input video file
├── output_video/           # Directory for output files
├── stubS/
│   └── tracks.pkl          # Cached tracking results


## Key Components

### 1. Object Detection (YOLO)
- Detects players, goalkeepers, referees, and ball
- Converts goalkeepers to player class for unified tracking
- Processes video in batches for efficiency

### Object Tracking (SORT)
- Assigns unique IDs to detected players
- Maintains track consistency across frames
- Handles occlusions and temporary disappearances

### Team Assignment
- Extracts player jersey colors from top half of bounding boxes
- Uses K-means clustering to identify dominant colors
- Groups players into two teams based on color similarity
- Assigns team colors for visualization



## Approach and Methodology
### System Architecture
- The system follows a modular pipeline approach:
    Detection Phase → Tracking Phase → Team Assignment → Visualization

    1. Object Detection: Utilizes YOLO (You Only Look Once) for real-time detection of players, referees, and ball
    2. Object Tracking: Implements SORT (Simple Online and Realtime Tracking) for maintaining consistent player identities
    3. Team Classification: Employs K-means clustering on jersey colors for automatic team assignment
    4. Visualization: Generates annotated video output with player IDs and team colors

### Technical Implementation
- Object Detection Strategy

    1. Model: Pre-trained YOLO model fine-tuned for football scenarios
    2. Classes: Players, goalkeepers, referees, and ball
    3. Optimization: Batch processing (20 frames) for computational efficiency
    4. Confidence Threshold: 0.5 for stable detections

### Tracking Algorithm
- SORT Implementation: Kalman filter-based tracking with Hungarian algorithm for data association
    Parameters:

    1. max_age=30: Maximum frames to maintain track without detection
    2. min_hits=3: Minimum detections required before track initialization
    3. iou_threshold=0.3: Intersection over Union threshold for matching



### Team Assignment Logic
- 
    1. Color Extraction: Analyzes top half of player bounding boxes (jersey area)
    2. Clustering: K-means with 2 clusters to identify team colors
    3. Background Removal: Filters out background pixels using corner cluster analysis
    4. Consistency: Maintains player-team mapping throughout video