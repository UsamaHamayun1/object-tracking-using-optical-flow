import os

import cv2


class frame_reader(object):
    """A class for reading frames from a video file and saving them as images.

    Args:
            location (str): The path to the video file.

    Attributes:
            location (str): The path to the video file.
            cam (cv2.VideoCapture): The video capture object.

    """

    def __init__(self, location):
        super(frame_reader, self).__init__()
        self.location = location
        self.cam = cv2.VideoCapture(location)

    def save_video(self, folder_name):
        """Extracts frames from the video and saves them as images in the specified folder.

        Args:
                folder_name (str): The name of the folder to save the frames in.

        """
        print("Current Extracting frames.")
        try:
            os.mkdir(folder_name)
        except:
            print("folder already there")
        current_frame = 0
        while True:
            ret, frame = self.cam.read()

            if ret:
                name_frame = str(current_frame)
                name = folder_name + "/%s" % name_frame + ".png"
                cv2.imwrite(name, frame)
                current_frame = current_frame + 1

            else:
                break

        self.cam.release()


if __name__ == "__main__":
    Easy = frame_reader("Easy.mp4")
    Easy.save_video("Easy")
