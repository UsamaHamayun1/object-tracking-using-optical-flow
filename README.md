# Optical Flow Object Tracking

This project implements optical flow for object tracking in videos.

## Project Structure

- `applyGeometricTransformations.py`: Applies geometric transformations to images or video frames.
- `estimateAllTranslation.py`: Estimates translations (movements) across a series of images or video frames.
- `estimatefeatureTranslation.py`: Estimates the translation of specific features within images or video frames.
- `frame_extractor.py`: Extracts frames from a video file.
- `getFeatures.py`: Identifies or extracts features from images or video frames.
- `object_tracker.py`: Tracks the movement of objects within a series of images or video frames.

## Installation

This project requires Python 3.8 and pip. You can install the required packages with:

```sh
pip install -r requirements.txt
```

## Running

A step by step series of examples that tell you how to get a development env running

Please follow code below to run this project:

```sh
python object_tracker.py 'video path' 'folder name for saving extracted images'
```

For example

```sh
python object_tracker.py Videos/Easy.mp4 Easy
```

## Steps

- After running the above code, it will ask for no. of objects to track. Enter the number of objects and hit space.
- Create a bounding box and hit space. Do the same process for each object.
- Computation will start and each frame will pop out for no. of features tracked in each frame.
- It will create bounding box for each object in its new position.
- Resulting Video will be saved in Results folder.
