# Train and Evaluate E2E-TAD on Your Dataset

## 1. Prepare data

- ActivityNet-style annotations:
  Our dataloader supports any dataset as long as the annotation file has the same format as ActivityNet. See the example
  below. The path of this annotation file is denoted as `ANNO_PATH`.

```JSON
{
  "database": {
    "video_id": {
      "duration": 12,
      "annotations": [
        {
          "label": "Futsal",
          "segment": [
            2.0,
            18.0
          ]
        }
      ]
    },
    "video_id2": {
    }
  }
}

```

- Video frames: Please refer to `tools/extract_frames.py` to extract video frames for your dataset. The root path of
  frames is denoted as `FRAME_PATH`. You should choose a proper FPS. If your dataset is similar to THUMOS14, you may
  extract frames at around 10 fps. If it is similar to ActivityNet, you may sample fixed number of frames from each
  video.

- Extra annotation file: Please refer to `tools/prepare_data.py` to generate a file that records the FPS and number of
  frame of each video. The path of this file is denoted as `FT_INFO_PATH`.

After these steps, please add the FRAME_PATH and FT_INFO_PATH info in `datasets/path.yml` for your dataset.

```
YOUR_DATASET:
  ann_file: ANNO_PATH
  img:     
    local_path: FRAME_PATH
    ft_info_file: FT_INFO_PATH
```

## 2. Modify code

- models/tadtr.py: modify the `build` function to specify the number of classes of your dataset.
- datasets/data_utils: modify the `get_dataset_info` function.
- datasets/tad_eval.py: modify line 66-72.
- engine.py: modify line 110.

## 3. Write a config file

Please refer to the existing config files.
You need to set some parameters. For example,

- slice_len: If the videos are long and the actions are short, you may need to cut videos into slices (windows). The
  slice_len should be set to a value such that most actions are shorter than the corresponding duration. (slice_len =
  slice_duration * fps)
- the number of queries: It should be set to a value that is slightly larger than the maximum number of actions per
  video.

## Badminton dataset
After the dataset is created, your project files structure should look like this:

```
├─ tools
│    ├─ extract_clips.py                      <-- full match video to clips with a specified length e.g. 180 seconds.
│    ├─ prepare_dataset.py                    <-- create badminton_annotations_with_fps_duration.json
│    ├─ extract_frames.py                     <-- clips to images with a specified fps
│    ├─ prepare_data.py                       <-- create badminton_img10fps_info.json (images count and video duration)
├─ data
    ├── badminton
        ├── raw
        │     ├── young_yamaguchi              <-- raw data used to create training data
        │     │   ├── young_yamaguchi.mp4
        │     │   ├── RallySeg.mp4
        │     ├── ginting_atonsen
        │     │    ├── ginting_atonsen.mp4
        │     │    ├── RallySeg.mp4  
        │     ├── ...  
        ├── videos
        │        ├── ginting_axelsen_000.mp4    <-- source clips ~180 seconds in 30 fps
        │        ├── ginting_axelsen_001.mp4
        │        ├── ...
        ├── img10fps 
        │    ├── ginting_axelsen_000            <-- sampled images in 10 fps 
        │       ├── img_0001.jpg                  
        │       ├── img_0002.jpg
        │       ├── ... 
        │    ├── ginting_axelsen_002
        │    ├── ...
        ├── badminton_img10fps_info.json        <-- annotation of feature length (number of images) and duration (clip duration in seconds)
        ├── badminton_annotations_with_fps_duration.json   <--  annotation of segements and labels for actions
```


## 5. Training and evaluation
Training and evaluation process is the same as THUMOS14.
