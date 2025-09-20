from video import VideoMaker


def main():
    video_maker = VideoMaker("MyVideo", "100712_sat.jpg")
    for i in range(150, 200):
        for j in range(150, 200):
            video_maker.change_pixel(i, j, (0, 255, 0))  # Change pixels to green
    video_maker.release()


if __name__ == "__main__":
    main()
