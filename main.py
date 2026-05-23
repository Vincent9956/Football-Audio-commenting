import glob
import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from penalty_area_detector import PenaltyAreaDetector
from debugger import Debugger


def _find_video():
    for p in sorted(glob.glob("input_videos/*")):
        if p.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            return p
    return None


def main():
    video_path = _find_video()
    if not video_path:
        raise FileNotFoundError("Kein Video in input_videos/ gefunden.")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.release()
    print("Video:", video_path, "| FPS:", round(video_fps, 3))

    video_frames = read_video(video_path)

    tracker = Tracker("models/best.pt")
    stub = "stubs/track_stubs.pkl"
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub)
    tracker.enrich_with_jersey_colors(tracks, video_frames, stub_path=stub)
    tracker.add_position_to_tracks(tracks)

    cam = CameraMovementEstimator(video_frames[0])
    cam_mv = cam.get_camera_movement(
        video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl"
    )
    cam.add_adjust_positions_to_tracks(tracks, cam_mv)

    pad = PenaltyAreaDetector()
    pa_data = pad.get_penalty_area_data(
        video_frames, read_from_stub=True, stub_path="stubs/penalty_area_stub.pkl"
    )
    pad.add_penalty_area_to_tracks(tracks, pa_data)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned != -1:
            tracks["players"][frame_num][assigned]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned].get("team", 0))
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    save_video(video_frames, "output_videos/output_video.avi", fps=video_fps)

    Debugger(video_path).generate(
        video_frames, tracks, team_ball_control,
        tracker, cam, cam_mv,
        "output_videos/debugging_video.avi", fps=video_fps,
    )


if __name__ == "__main__":
    main()
