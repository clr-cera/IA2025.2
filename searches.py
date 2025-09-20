import os
from video import VideoMaker
import heapq
import cv2
import numpy as np
from enum import Enum


class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    # VISITED = (247, 154, 187)
    # PATH = (104, 175, 224)
    # BLOCKED = (142, 118, 247)
    VISITED = GREEN
    PATH = BLUE
    BLOCKED = RED


def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right


def bfs(origin, target, image_num):
    # O vídeo usa a imagem satélite como fundo
    video_maker = VideoMaker("bfs_Visualization", image_num + "_sat.jpg")
    # A máscara é usada para determinar onde pode ou não andar
    image = cv2.imread(os.path.join("images", image_num + "_mask.png"))
    queue = [origin]

    while queue:
        current = queue.pop(0)
        if current == target:
            break
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            # Se cai fora da imagem, ignora
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                # Se é branco (caminho livre) e não foi visitado ainda
                if np.array_equal(image[neighbor[1], neighbor[0]], Color.WHITE.value):
                    queue.append(neighbor)
                    image[neighbor[1], neighbor[0]] = (
                        Color.VISITED.value
                    )  # Mark as visited
                    video_maker.change_pixel(
                        neighbor[0], neighbor[1], Color.VISITED.value
                    )
                # Se é preto (bloco), marca como bloqueado em vermelhinho no vídeo
                elif np.array_equal(image[neighbor[1], neighbor[0]], Color.BLACK.value):
                    video_maker.change_pixel(
                        neighbor[0], neighbor[1], Color.BLOCKED.value
                    )
    # Finaliza o vídeo
    video_maker.release()


def astar(origin, target, image_num):
    # O vídeo usa a imagem satélite como fundo
    video_maker = VideoMaker("AStar_Visualization", image_num + "_sat.jpg")
    # A máscara é usada para determinar onde pode ou não andar
    image = cv2.imread(os.path.join("images", image_num + "_mask.png"))
    queue = []
    heapq.heappush(queue, (0, origin))
    visited_by = {origin: None}
    # g_score guarda o custo do caminho mais barato até um nó
    # f_score guarda o custo estimado do caminho mais barato até o alvo passando por um nó
    g_score = {origin: 0}
    f_score = {origin: distance(origin, target)}
    while queue:
        current = heapq.heappop(queue)[1]
        if current == target:
            break
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Se cai fora da imagem, ignora
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                # Se é branco (caminho livre) e não foi visitado ainda
                if np.array_equal(image[neighbor[1], neighbor[0]], Color.WHITE.value):
                    tentative_g_score = g_score[current] + 1
                    # Se esse caminho até o vizinho é melhor, registra ele
                    if tentative_g_score < g_score.get(neighbor, float("inf")):
                        visited_by[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + distance(
                            neighbor, target
                        )
                        heapq.heappush(queue, (f_score[neighbor], neighbor))
                        image[neighbor[1], neighbor[0]] = (
                            Color.VISITED.value
                        )  # Mark as visited
                        video_maker.change_pixel(
                            neighbor[0], neighbor[1], Color.VISITED.value
                        )
                elif np.array_equal(image[neighbor[1], neighbor[0]], Color.BLACK.value):
                    video_maker.change_pixel(
                        neighbor[0], neighbor[1], Color.BLOCKED.value
                    )
    # Reconstrói o caminho
    pathnode = target
    while pathnode != None:
        video_maker.change_pixel(pathnode[0], pathnode[1], Color.PATH.value)
        # To make the path more visible, color all neighbors too
        for direction in DIRECTIONS:
            neighbor = (pathnode[0] + direction[0], pathnode[1] + direction[1])
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                video_maker.change_pixel(neighbor[0], neighbor[1], Color.PATH.value)
        pathnode = visited_by.get(pathnode, None)
    video_maker.release()
