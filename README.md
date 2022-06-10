# Running Detections

This is based of of previous code written for YOLOv5 with DeepSort modified in it. To run predictions, store a video in .mp4 format in the root directory and run

```
python tracker.py --source vid
```
where vid is the name of the file with the .mp4 ending. Results will be stored in the root directory with the ending `vid_results.mp4` where vid is the original name