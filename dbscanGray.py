import cv2 as cv
import numpy as np
import math
import time


d = 1  # channels


def euclideanDistance(pointA, pointB):
    channelsDifferencePower = math.pow(int(pointA) - int(pointB), 2)
    distance = math.sqrt(channelsDifferencePower)
    return distance


def getNeighborPts(img, point, eps, center):
    # print('Here4')
    neighborPts = [[col, row]
                   for row in range(point[1] - 1, point[1] + 2)
                   for col in range(point[0] - 1, point[0] + 2)
                   if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]
                   if euclideanDistance(img[row][col], center) <= eps]
    return neighborPts


def dbscanClustering(src, eps, minPts):
    print("Start: %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    cv.imshow('Original', src)
    img = np.copy(src)
    rows = img.shape[0]
    cols = img.shape[1]
    visitedPts = np.zeros((rows, cols))
    hasCluster = np.zeros((rows, cols))
    clusters = []
    label = 0
    for row in range(rows):
        for col in range(cols):
            label += 1
            if visitedPts[row][col] != 0:
                continue
            visitedPts[row][col] = label
            gray_mean = 0
            point = [col, row]
            center = img[point[1]][point[0]]
            neighbourPts = getNeighborPts(img, point, eps, center)
            if len(neighbourPts) < minPts:
                visitedPts[point[1]][point[0]] = -1 # MARK as NOISE
                continue
            newCluster = [point]
            cnt = 0

            for neighbourPt in neighbourPts:
                cnt += 1
                if visitedPts[neighbourPt[1]][neighbourPt[0]] == 0:
                    visitedPts[neighbourPt[1]][neighbourPt[0]] = label
                    newNeighbours = getNeighborPts(img, neighbourPt, eps, center)
                    if len(newNeighbours) >= minPts:
                        neighbourPts.extend(newNeighbours)
                if hasCluster[neighbourPt[1]][neighbourPt[0]] == 0:
                    newCluster.append(neighbourPt)
                    gray_mean += img[neighbourPt[1]][neighbourPt[0]]
                    hasCluster[neighbourPt[1]][neighbourPt[0]] = 1
            centroid = newCluster[0]
            img[centroid[1]][centroid[0]] = gray_mean / len(newCluster)
            clusters.append(newCluster)
    for i in range(len(clusters)):
        currentCluster = clusters[i]
        firstPoint = currentCluster[0]
        for point in currentCluster:
            img[point[1]][point[0]] = np.copy(img[firstPoint[1]][firstPoint[0]])
    exec_time = time.time() - start_time
    print("\tSecondi: %s" % exec_time)
    print("Finish: %s" % (time.asctime(time.localtime(time.time()))))

    return [np.copy(img), str(exec_time)]
