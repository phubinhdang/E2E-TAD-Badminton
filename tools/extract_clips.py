import argparse
import os
import subprocess
from pathlib import Path


def extract_clips(input_video, output_dir):
    input_video = Path(input_video)
    assert input_video.exists()
    video_name = Path(input_video).stem
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.mkdir(output_dir)
    command = [
        "ffmpeg",
        "-i", f"{input_video}",
        "-c", "copy",
        "-map", "0",
        "-force_key_frames", "expr:gte(t,n_forced*180)",
        "-segment_time", "180",
        "-f", "segment",
        "-reset_timestamps", "1",
        f"{output_dir}/{video_name}_%03d.mp4"
    ]

    subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True,
                        help='Path to input video, relative to script working dir')
    # FIXME: should later change to data/badminton/clips
    parser.add_argument('--output_dir', type=str, default='data/badminton/videos',
                        help='Dir for the output clips, relative to script working dir')
    parser.add_argument('--clip_len', type=int, default=180, help='Duration of clip in seconds')
    args = parser.parse_args()
    extract_clips(args.input_video, args.output_dir)
