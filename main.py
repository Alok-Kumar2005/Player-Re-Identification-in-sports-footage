from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    video_frames = read_video('video/15sec_input_720p.mp4')

    tracker = Tracker('model/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubS/tracks.pkl')

    ## Saving image for a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]
    #     x1, y1, x2, y2 = map(int, bbox)
    #     player_image = frame[y1:y2, x1:x2]
    #     cv2.imwrite(f'output_video/player_image2.jpg', player_image)
    #     break

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    


    ## save video
    save_video(output_video_frames, 'output_video/output_video.avi', fps=24)


if __name__ == "__main__":
    main()