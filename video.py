import cv2
import numpy as np
import os


class VideoMaker:
    def __init__(self, title, image_path):
        self.title = title
        self.frame = cv2.imread(os.path.join("images", image_path))
        self.video = cv2.VideoWriter(
            f"{self.title}.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            30,
            (self.frame.shape[1], self.frame.shape[0]),
        )
        self.video.write(self.frame)

    def change_pixel(self, x, y, color):
        print(f"Changing pixel at ({x}, {y}) to color {color}")
        self.frame[y, x] = color
        self.video.write(self.frame)

    def release(self):
        self.video.release()
        cv2.destroyAllWindows()
