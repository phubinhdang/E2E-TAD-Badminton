import argparse
import concurrent.futures
import json
import os
import os.path as osp
from pathlib import Path

from tqdm import tqdm


def extract_frames(video_path, dst_dir, fps):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    video_fname = osp.basename(video_path)

    if not osp.exists(video_path):
        subdir = 'test_set/TH14_test_set_mp4' if 'test' in video_fname else 'Validation_set/videos'
        url = f'https://crcv.ucf.edu/THUMOS14/{subdir}/{video_fname}'
        os.system('wget {} -O {} --no-check-certificate'.format(url, video_path))
    # ffmpeg with a progres bar
    cmd = 'ffpb -i "{}"  -filter:v "fps=fps={}" "{}/img_%07d.jpg"'.format(video_path, fps, dst_dir)

    # Redirect output to /dev/null (Unix) or NUL (Windows)
    if os.name == 'posix':  # Unix-like OS
        ret_code = os.system(cmd + ' > /dev/null 2>&1')
    else:  # Windows
        ret_code = os.system(cmd + ' > NUL 2>&1')

    if ret_code == 0:
        os.system('touch logs/frame_extracted_{}fps/{}'.format(fps, osp.splitext(osp.basename(video_path))[0]))
    return ret_code == 0


def parse_args():
    parser = argparse.ArgumentParser('Extract frames')
    parser.add_argument('--data_dir', help='should contains "videos" subdir and an annotation file', required=True)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('-s', '--start', type=int)
    parser.add_argument('-e', '--end', type=int)

    args = parser.parse_args()
    return args


def mkdir_if_missing(dirname):
    if not osp.exists(dirname):
        os.makedirs(dirname)


def main(subset, should_continue=False):
    args = parse_args()

    log_dir = 'logs/frame_extracted_{}fps'.format(args.fps)
    mkdir_if_missing(log_dir)

    video_dir = Path(args.data_dir) / "videos"
    frame_dir = Path(args.data_dir) / f"img{args.fps}fps"
    mkdir_if_missing(frame_dir)

    annotation_file_path = Path(args.data_dir) / "badminton_annotations_with_fps_duration.json"
    database = json.load(open(annotation_file_path))['database']
    vid_names = list(sorted([x for x in database if database[x]['subset'] == subset]))

    start_ind = 0 if args.start is None else args.start
    end_ind = len(vid_names) if args.end is None else min(args.end, len(vid_names))

    vid_names = vid_names[start_ind:end_ind]

    if should_continue:
        finished = os.listdir(log_dir)
        videos_todo = list(sorted(set(vid_names).difference(finished)))
    else:
        videos_todo = vid_names

    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        # Use tqdm to create a progress bar
        futures = {executor.submit(extract_frames, video_dir / f"{x}.mp4",
                                   frame_dir / x, args.fps): x for x in videos_todo}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Extracting frames: '):
            # You can optionally check if the future was successful
            try:
                future.result()  # This will raise an exception if the function failed
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")


if __name__ == '__main__':
    main('val')
    main('test')

# thumos14 size is 78GB, see https://github.com/open-mmlab/mmaction2/blob/main/tools/data/thumos14/download_videos.sh
# python tools/extract_frames.py --data_dir data/inference --fps 10 -s 1 -e 2

