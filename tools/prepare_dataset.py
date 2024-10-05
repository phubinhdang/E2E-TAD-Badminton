import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict

import pandas
from pydantic import BaseModel
from typing_extensions import Literal


class Segment(BaseModel):
    segment: List[float]
    label: str


class ClipAnnotation(BaseModel):
    subset: str
    annotations: List[Segment]
    fps: float
    duration: float


class Match(BaseModel):
    name: str
    duration_in_seconds: float
    fps: int
    num_extracted_clips: int
    subset: Literal["val", "test"]


def get_video_duration_in_seconds(video_file):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return float(result.stdout.strip())


def get_fps(video_file):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # extracting, for example 30 from 30/1
    return int(result.stdout.split("/")[0])


def generate_clips_segments_for_match(match_name, rally_seg_path: Path, fps, num_clips, clip_dir, clip_len,
                                      subset: Literal["val", "test"], offset_correction, inference_mode: bool) -> Dict:
    print(f"Using offset of {offset_correction} seconds for correction")
    clip_annos = {}
    # find rally segments in each clip
    for i in range(num_clips):
        clip_name = f'{match_name}_{i:03d}'
        clip_dir = Path(clip_dir)
        clip_duration = get_video_duration_in_seconds(clip_dir / f'{clip_name}.mp4')
        if inference_mode:
            clip_annos[clip_name] = ClipAnnotation(
                subset=subset,
                annotations=[],
                fps=fps,
                duration=clip_duration
            ).model_dump()
        else:
            # for each rally annotation in the whole match for training
            df_rally_seg = pandas.read_csv(rally_seg_path)
            df_rally_seg['Start_Second'] = (df_rally_seg['Start'] / fps).round(2)
            df_rally_seg['End_Second'] = (df_rally_seg['End'] / fps).round(2)

            df_rally_seg = df_rally_seg[['Start_Second', 'End_Second']]
            segment_start_time = i * clip_len
            segment_end_time = (i + 1) * clip_len
            segments = []
            for _, row in df_rally_seg.iterrows():
                start_second = row['Start_Second'] + offset_correction  # Apply offset correction
                end_second = row['End_Second'] + offset_correction  # Apply offset correction

                # Check if the annotation falls within this segment
                if end_second > segment_start_time and start_second < segment_end_time:
                    # Calculate the annotation relative to the start of the segment
                    start_time = max(0, start_second - segment_start_time)
                    end_time = min(clip_len, end_second - segment_start_time)
                    segments.append(Segment(
                        segment=[round(start_time, 1), round(end_time, 1)],
                        label="rally"
                    ))
            if segments:
                clip_annos[clip_name] = ClipAnnotation(
                    subset=subset,
                    annotations=segments,
                    fps=fps,
                    duration=clip_duration
                ).model_dump()

    return clip_annos


def write_to_file(clip_segments, file_name):
    with open(file_name, 'w') as f:
        json.dump(clip_segments, f, indent=4)
    print("Wrote clip segment annotations to file: ", file_name)


def count_extracted_clips(clip_dir: Path, match_name) -> int:
    count = 0
    for clip in clip_dir.iterdir():
        if match_name in str(clip):
            count = count + 1
    return count


def get_matches(data_dir: Path, test_match: str, clip_dir: Path, inference_mode: bool) -> List[Match]:
    data_dir = data_dir / 'raw'
    matches = []
    for match_dir in data_dir.iterdir():
        match_name = str(match_dir).split("/")[-1]
        if inference_mode or match_name == test_match:
            subset = "test"
        else:
            subset = "val"
        match_video_file = match_dir / f'{match_name}.mp4'
        match = Match(name=match_name,
                      duration_in_seconds=get_video_duration_in_seconds(match_video_file),
                      fps=get_fps(match_video_file),
                      num_extracted_clips=count_extracted_clips(clip_dir, match_name),
                      subset=subset)
        matches.append(match)
    return matches


def prepare_dataset(data_dir: str, test_match: str, clip_len: float, offset_correction: float,
                    inference_mode: bool):
    data_dir = Path(data_dir)
    clip_dir = data_dir / 'videos'
    assert clip_dir.is_dir()
    matches: List[Match] = get_matches(Path(data_dir), test_match, clip_dir, inference_mode)
    database = {}
    print(f"Creating dataset for {'INFERENCE' if inference_mode else 'TRAINING'}")
    for m in matches:
        print("Processing match: ", m)
        rally_seg_path = None if inference_mode else Path(data_dir) / 'raw' / m.name / 'RallySeg.csv'
        # TODO: match and clips segment annotation still not line up perfectly,
        #  specially in rallies at the end of the match, please refine the clip segments generation
        if m.name == "ginting_antonsen":
            offset_correction = -4
        clips_segments = generate_clips_segments_for_match(m.name, rally_seg_path, m.fps, m.num_extracted_clips,
                                                           clip_dir,
                                                           clip_len,
                                                           m.subset,
                                                           offset_correction,
                                                           inference_mode)

        database.update(clips_segments)
    write_to_file({'database': database}, data_dir / 'badminton_annotations_with_fps_duration.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # FIXME: should later change to data/badminton/clips
    parser.add_argument('--data_dir', type=str, default='data/badminton',
                        help='Dir that stores that extracted clips from full match videos')
    parser.add_argument('--test_match', type=str,
                        help='match name is used for test, the rest is automatically used for training')
    parser.add_argument("--clip_len", type=float, default=180,
                        help="Clip length in seconds, it must be equal to the value used for extracting clips")
    parser.add_argument("--offset_correction", type=float, default=-1,
                        help="Number of seconds to align the clips segment annotations with that of the match")
    parser.add_argument("--inference", action="store_true",
                        help="create dataset for inferences, if not specified, it will be training model")
    args = parser.parse_args()
    prepare_dataset(args.data_dir, args.test_match, args.clip_len, args.offset_correction, args.inference)
