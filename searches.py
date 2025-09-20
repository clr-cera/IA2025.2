import os
from video import VideoMaker
import heapq
import cv2
import numpy as np


def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def bfs(origin, target, image_num):
    video_maker = VideoMaker("AStar_Visualization", image_num + "_sat.jpg")
    image = cv2.imread(os.path.join("images", image_num + "_mask.png"))
    queue = [origin]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        current = queue.pop(0)
        if current == target:
            break
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                if np.array_equal(image[neighbor[1], neighbor[0]], [255, 255, 255]):
                    queue.append(neighbor)
                    video_maker.change_pixel(neighbor[0], neighbor[1], (0, 255, 0))
    video_maker.release()


def astar(origin, target, image_num):
    video_maker = VideoMaker("AStar_Visualization", image_num + "_sat.jpg")
    image = cv2.imread(os.path.join("images", image_num + "_mask.png"))
    queue = []
    heapq.heappush(queue, (0, origin))
    g_score = {origin: 0}
    f_score = {origin: distance(origin, target)}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    while queue:
        current = heapq.heappop(queue)[1]
        if current == target:
            break
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                if np.array_equal(image[neighbor[1], neighbor[0]], [255, 255, 255]):
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float("inf")):
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + distance(
                            neighbor, target
                        )
                        heapq.heappush(queue, (f_score[neighbor], neighbor))
                        video_maker.change_pixel(neighbor[0], neighbor[1], (0, 255, 0))
    video_maker.release()
