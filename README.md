# video-surveillance
A smart video surveillance system with motion detection, person tracking, and alert generation. Small project made for the Fundamentals of Computer Vision lab during my first year of Machine Learning master studies at Polytechnic University of Timișoara.

## Running the project

Install the required libraries:
- OpenCV
- NumPy
- ultralytics
- scipy
- requests

Then run:

`python main.py --config <config_path>`

Replace `<config_path>` with the path to your JSON configuration file.

Optionally, you can select a Region of Interest (RoI) if that mode is enabled in the config.

## Configuration file structure

The config is a `.json` file that looks like this:

```json
{
  "video_path": "input.mp4",
  "camera_mode": false,
  "roi_mode": false,
  "save_detections": true,
  "save_motion_video": true,
  "motion_detector": { ... },
  "tracker": { ... },
  "movement_classification": { ... },
  "door_detection": { ... },
  "heatmap": { ... }
}
```

Each section lets you customize how the system behaves, including motion detection, tracking, movement classification, door alerts, and heatmap generation.

Door alerts are done with Pushover. For notifications to work, you must modify the `notifications.py` file with your own Pushover API credentials and have the app installed on your device.

## Modes

- **Full Mode**: Detects and classifies human movement using YOLO and a tracker. It captures snapshots, builds motion videos, sends door alerts, and generates heatmaps.
- **RoI Mode**: Focuses only on motion detection in a selected region. Skips person detection and other features for performance and simplicity.

## Documentation

Read more about this project at the [documentation](https://github.com/efrem-upt/video_surveillance/blob/main/docs/video_surveillance_documentation.pdf).

## License

[MIT](https://choosealicense.com/licenses/mit/)
