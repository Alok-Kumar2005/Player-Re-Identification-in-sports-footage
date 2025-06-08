from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('video/15sec_input_720p.mp4')

    tracker = Tracker('model/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubS/tracks.pkl')

    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    ## save video
    save_video(output_video_frames, 'output_video/output_video.avi', fps=24)


if __name__ == "__main__":
    main()