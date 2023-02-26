import numpy as np
import math
import cv2 as cv


minRegAreaFactor = 0.0095
pointShift = ((-1, -1),
              (-1, 0),
              (-1, 1),
              (0, -1),
              (0, 1),
              (1, -1),
              (1, 0),
              (1, 1))


def colordistance(color_a, color_b):
    blue = int(color_a[0]) - int(color_b[0])
    green = int(color_a[1]) - int(color_b[1])
    red = int(color_a[2]) - int(color_b[2])
    distance = math.sqrt((blue ** 2) + (green ** 2) + (red ** 2))
    return distance


def grow(img, labeledImg, mask, seedPt, colorDistance, label, copyLabImg, copyIgnoredImg):
    pointStack = [seedPt]
    clusterPts = [seedPt]
    b = img[seedPt[1]][seedPt[0]][0]
    g = img[seedPt[1]][seedPt[0]][1]
    r = img[seedPt[1]][seedPt[0]][2]
    while len(pointStack) > 0:
        centerPt = pointStack.pop()
        copyLabImg[centerPt[1]][centerPt[0]] = label
        copyIgnoredImg[centerPt[1]][centerPt[0]] = 255
        mask[centerPt[1]][centerPt[0]] = 1
        for i in range(8):
            currPt = [0, 0]
            currPt[0] = centerPt[0] + pointShift[i][0]
            currPt[1] = centerPt[1] + pointShift[i][1]
            if 0 <= currPt[1] < img.shape[0] and 0 <= currPt[0] < img.shape[1]:
                dist = colordistance(img[currPt[1]][currPt[0]], img[seedPt[1]][seedPt[0]])
                if dist <= colorDistance and labeledImg[currPt[1]][currPt[0]] == 0 \
                        and mask[currPt[1]][currPt[0]] == 0:
                    copyLabImg[currPt[1]][currPt[0]] = label
                    copyIgnoredImg[currPt[1]][currPt[0]] = 255
                    mask[currPt[1]][currPt[0]] = 1
                    pointStack.append(currPt)
                    clusterPts.append(currPt)
                    b += int(img[currPt[1]][currPt[0]][0])
                    g += int(img[currPt[1]][currPt[0]][1])
                    r += int(img[currPt[1]][currPt[0]][2])
    b /= len(clusterPts)
    g /= len(clusterPts)
    r /= len(clusterPts)
    colorMean = [b, g, r]
    return [clusterPts, colorMean]


def regionMean(img, clusterPts, colorMean):
    for pt in clusterPts:
        img[pt[1]][pt[0]][0] = colorMean[0]
        img[pt[1]][pt[0]][1] = colorMean[1]
        img[pt[1]][pt[0]][2] = colorMean[2]


def labeling(src, colorDistance):
    img = np.copy(src)
    label = 1
    minRegArea = int(img.shape[0] * img.shape[1] * minRegAreaFactor)
    labeledImg = np.zeros((img.shape[0], img.shape[1]))
    mask = np.copy(labeledImg)
    clusters = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if labeledImg[i][j] == 0:
                seedPt = [j, i]
                copyLabeledImg = np.copy(labeledImg)
                copyIgnoredImg = np.copy(labeledImg)
                clusterPts, colorMean = grow(img, labeledImg, mask, seedPt, colorDistance, label, copyLabeledImg, copyIgnoredImg)
                clusters.append(clusterPts)
                # regionMean(img, clusterPts, colorMean)
                regionArea = np.sum(mask)
                if regionArea > minRegArea:
                    '''print('//////////BEFORE//////////')
                    print(labeledImg)'''
                    # print('NewLabel', label)
                    # labeledImg = [[x + (y * label) for x, y in zip(row1, row2)] for row1, row2 in zip(labeledImg, mask)]
                    labeledImg = np.copy(copyLabeledImg)
                    '''print('//////////AFTER//////////')
                    print(labeledImg)'''
                    # labeledImg += mask * label
                    label += 1
                    '''cv.imshow('region' + str(label), mask * 255)
                    cv.waitKey(0)'''
                else:
                    # print('Ignored')
                    labeledImg = np.copy(copyIgnoredImg)
                    # labeledImg = [[x + (y * 255) for x, y in zip(row1, row2)] for row1, row2 in zip(labeledImg, mask)]
            mask.fill(0)
    print('TotClusters: ', label)
    labels = []
    for i in range(label):
        labels.append(i)
    return [img, labeledImg, label]