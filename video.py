import cv2
import os

IGNOREDFRAMES = 100


class VideoMaker:
    def __init__(self, title, image_path):
        self.title = title
        self.frame = cv2.imread(os.path.join("images", image_path))
        self.video = cv2.VideoWriter(
            f"{self.title}.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            64,
            (self.frame.shape[1], self.frame.shape[0]),
        )
        self.video.write(self.frame)
        self.changes_count = 0

    def change_pixel(self, x, y, color):
        self.frame[y, x] = color
        self.changes_count += 1
        if self.changes_count % IGNOREDFRAMES == 0:
            self.video.write(self.frame)

    def release(self):
        self.video.release()
        cv2.destroyAllWindows()
