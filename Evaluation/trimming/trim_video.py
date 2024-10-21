import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
import argparse
from pathlib import Path
import shutil


def extract_clip(input_video, start_time, end_time, output_video):
    # Load the video
    video = VideoFileClip(input_video)

    # Extract the subclip (moviepy uses seconds, so we convert hh:mm:ss to seconds)
    subclip = video.subclip(start_time, end_time)

    # Write the result to a file
    subclip.write_videofile(output_video, codec="libx264")


def combine_clips(clips_list, output_video):
    # List to store VideoFileClip objects
    video_clips = []

    # Load each clip from the list and append to video_clips list
    for clip_path in clips_list:
        clip = VideoFileClip(clip_path)
        video_clips.append(clip)

    # Concatenate the list of video clips
    final_clip = concatenate_videoclips(video_clips, method="compose")

    # Write the final video to an output file
    final_clip.write_videofile(output_video, codec="libx264")

    # Close all the clips
    for clip in video_clips:
        clip.close()


def generate_summary_clip(data_dir: str, output_video_name: str):
    data_dir = Path(data_dir)
    print(data_dir)
    assert data_dir.exists()

    match_name = data_dir.name
    df = pd.read_csv(data_dir / "clips.csv")
    input_video = str(data_dir / f"{match_name}.mp4")
    subclip_dir = data_dir / "subclips"
    if subclip_dir.exists():
        shutil.rmtree(subclip_dir)
        print("Removed existing subclips before creating new ones")
    subclip_dir.mkdir(parents=True, exist_ok=True)

    subclip_names = []
    for i, (s, e) in enumerate(zip(df["start_hhmmss"], df["end_hhmmss"])):
        # Example usage
        output_video = f"{subclip_dir}/{i}.mp4"
        extract_clip(input_video, s, e, output_video)
        subclip_names.append(output_video)
    output_video_path = str(data_dir / output_video_name)
    combine_clips(subclip_names, output_video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to input video and the rally prediction info",
    )
    parser.add_argument(
        "--output_video_name",
        type=str,
        default="summary.mp4",
    )
    args = parser.parse_args()
    generate_summary_clip(args.data_dir, args.output_video_name)
