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
    visited_by = {origin: None}
    image[origin[1], origin[0]] = Color.VISITED.value

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
                    visited_by[neighbor] = current
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

    # Reconstrói o caminho
    pathnode = target
    custo_caminho = 1
    video_maker.IGNOREDFRAMES = 16
    while pathnode != None:
        video_maker.change_pixel(pathnode[0], pathnode[1], Color.PATH.value)
        # To make the path more visible, color all neighbors too
        for direction in DIRECTIONS:
            neighbor = (pathnode[0] + direction[0], pathnode[1] + direction[1])
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                video_maker.change_pixel(neighbor[0], neighbor[1], Color.PATH.value)
        pathnode = visited_by.get(pathnode, None)
        custo_caminho += 1

    # Finaliza o vídeo
    video_maker.release()
    print("Custo do caminho BFS: ", custo_caminho)


def dfs(origin, target, image_num):
    # O vídeo usa a imagem satélite como fundo
    video_maker = VideoMaker("dfs_Visualization", image_num + "_sat.jpg")
    # A máscara é usada para determinar onde pode ou não andar
    image = cv2.imread(os.path.join("images", image_num + "_mask.png"))
    stack = [origin]
    visited_by = {origin: None}
    cost = {origin: 0}
    image[origin[1], origin[0]] = Color.VISITED.value

    while stack:
        current = stack.pop(-1)
        if current == target:
            break
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            # Se cai fora da imagem, ignora
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                # Se é branco (caminho livre) e não foi visitado ainda
                if np.array_equal(image[neighbor[1], neighbor[0]], Color.WHITE.value):
                    stack.append(neighbor)

                    cost[neighbor] = cost[current] + 1
                    visited_by[neighbor] = current

                    image[neighbor[1], neighbor[0]] = (
                        Color.VISITED.value
                    )  # Mark as visited
                    video_maker.change_pixel(
                        neighbor[0], neighbor[1], Color.VISITED.value
                    )
                elif np.array_equal(
                    image[neighbor[1], neighbor[0]], Color.VISITED.value
                ):
                    if cost.get(neighbor, float("inf")) > cost[current] + 1:
                        cost[neighbor] = cost[current] + 1
                        visited_by[neighbor] = current

                # Se é preto (bloco), marca como bloqueado em vermelhinho no vídeo
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

    image[origin[1], origin[0]] = Color.VISITED.value

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
                if np.array_equal(image[neighbor[1], neighbor[0]], Color.VISITED.value):
                    tentative_g_score = g_score[current] + 1
                    # Se esse caminho até o vizinho é melhor, registra ele
                    if tentative_g_score < g_score.get(neighbor, float("inf")):
                        visited_by[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + distance(
                            neighbor, target
                        )
                        heapq.heappush(queue, (f_score[neighbor], neighbor))
                elif np.array_equal(image[neighbor[1], neighbor[0]], Color.BLACK.value):
                    video_maker.change_pixel(
                        neighbor[0], neighbor[1], Color.BLOCKED.value
                    )
    # Reconstrói o caminho
    pathnode = target
    video_maker.IGNOREDFRAMES = 16
    caminho_custo = 1
    while pathnode != None:
        video_maker.change_pixel(pathnode[0], pathnode[1], Color.PATH.value)
        # To make the path more visible, color all neighbors too
        for direction in DIRECTIONS:
            neighbor = (pathnode[0] + direction[0], pathnode[1] + direction[1])
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                video_maker.change_pixel(neighbor[0], neighbor[1], Color.PATH.value)
        pathnode = visited_by.get(pathnode, None)
        caminho_custo += 1
    video_maker.release()
    print("Custo do caminho A*: ", caminho_custo)


def hill_climbing(origin, target, image_num):
    # O vídeo usa a imagem satélite como fundo
    video_maker = VideoMaker("HillClimbing_Visualization", image_num + "_sat.jpg")
    # A máscara é usada para determinar onde pode ou não andar
    image = cv2.imread(os.path.join("images", image_num + "_mask.png"))
    current = origin
    visited_by = {origin: None}
    image[origin[1], origin[0]] = Color.VISITED.value

    while current != target:
        neighbors = []
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            # Se cai fora da imagem, ignora
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                # Se é branco (caminho livre) e não foi visitado ainda
                if np.array_equal(image[neighbor[1], neighbor[0]], Color.WHITE.value):
                    neighbors.append(neighbor)
        if not neighbors:
            break  # Sem mais vizinhos para explorar
        # Escolhe o vizinho que está mais próximo do alvo
        next_node = min(neighbors, key=lambda n: distance(n, target))
        if distance(next_node, target) >= distance(current, target):
            raise Exception("NoPath")
            # Nenhum progresso possível
        visited_by[next_node] = current
        current = next_node
        image[current[1], current[0]] = Color.VISITED.value
        video_maker.change_pixel(current[0], current[1], Color.VISITED.value)

    # Reconstrói o caminho
    pathnode = current
    video_maker.IGNOREDFRAMES = 16  # To make the path drawing faster
    while pathnode != None:
        video_maker.change_pixel(pathnode[0], pathnode[1], Color.PATH.value)
        # To make the path more visible, color all neighbors too
        for direction in DIRECTIONS:
            neighbor = (pathnode[0] + direction[0], pathnode[1] + direction[1])
            if 0 <= neighbor[0] < image.shape[1] and 0 <= neighbor[1] < image.shape[0]:
                video_maker.change_pixel(neighbor[0], neighbor[1], Color.PATH.value)
        pathnode = visited_by.get(pathnode, None)
    video_maker.release()
