import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel
from typing_extensions import Literal


class Segment(BaseModel):
    segment: List[float]
    label: str

class Match(BaseModel):
    name: str
    subset: Literal["val", "test"]
    segments: List[Segment]
    fps: int
    duration_in_seconds: float


def get_video_duration_in_seconds(video_file):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return float(result.stdout.strip())


def get_fps(video_file):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # extracting, for example 30 from 30/1
    return int(result.stdout.split("/")[0])


def write_to_file(clip_segments, file_name):
    with open(file_name, "w") as f:
        json.dump(clip_segments, f, indent=4)
    print("Wrote clip segment annotations to file: ", file_name)


def get_segments(match_rally_seg, fps) -> List[Segment]:
    df = pd.read_csv(match_rally_seg)
    df['Start_Second'] = round(df['Start']/fps, 2)
    df['End_Second'] = round(df['End']/fps, 2)
    segments = []
    for s, e in zip(df['Start_Second'], df['End_Second']):
        segments.append(Segment(segment = [s, e], label="rally"))
    return segments

def get_matches(data_dir: Path, test_match: str) -> List[Match]:
    data_dir = data_dir / "raw"
    matches = []
    for match_dir in data_dir.iterdir():
        match_name = str(match_dir).split("/")[-1]
        if match_name == test_match:
            subset = "test"
        else:
            subset = "val"
        match_video_file = match_dir / f"{match_name}.mp4"
        fps = get_fps(match_video_file)
        match = Match(
            name=match_name,
            subset=subset,
            segments=get_segments(match_dir / "RallySeg.csv", fps),
            fps=fps,
            duration_in_seconds=get_video_duration_in_seconds(match_video_file),
        )
        matches.append(match)
    return matches


def prepare_dataset(data_dir: str, test_match: str):
    data_dir = Path(data_dir)
    assert data_dir.is_dir()
    matches: List[Match] = get_matches(Path(data_dir), test_match)
    database = {}
    for m in matches:
        annotations = []
        for seg in m.segments:
            annotations.append(seg.model_dump())
        database[m.name] = {
            "subset": m.subset,
            "annotations": annotations,
            "fps": m.fps,
            "duration": m.duration_in_seconds
        }
    write_to_file(
        {"database": database},
        data_dir / "badminton_annotations_with_fps_duration.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # FIXME: should later change to data/badminton/clips
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/badminton",
        help="Dir that stores that extracted clips from full match videos",
    )
    parser.add_argument(
        "--test_match",
        type=str,
        help="match name is used for test, the rest is automatically used for training",
    )
    args = parser.parse_args()
    prepare_dataset(args.data_dir, args.test_match)
