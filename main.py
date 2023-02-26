import cv2 as cv
import numpy as np
import dbscan
import dbscanGray
import regionGrowing as rg
from scipy import io
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score


def main():
    '''wantImgGray = True
    imagesPath = '../images/'
    imagesDirs = [f for f in listdir(imagesPath)]
    for imagesDir in imagesDirs:
        imagesDir = join(imagesPath, imagesDir)
        images = [f for f in listdir(imagesDir) if isfile(join(imagesDir, f))]
        for image_name in images:
            image_path = join(imagesDir, image_name)
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            color_string = 'color'
            if wantImgGray:
                color_string = 'gray'
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # img = np.stack((img,) * 3, axis=-1)

            img = cv.resize(img, (320, 240))
            cv.imshow('Original', img)
            cv.waitKey(0)
            repeat = 1

            while repeat != 0:
                eps = int(input('Enter range of color: '))
                minPts = int(input('Enter minPts (max 9 in a tridim-array of 3x3 points): '))
                # minPts = 9
                if len(img.shape) > 2:
                    mspp_img = dbscan.dbscanClustering(img, eps, minPts)
                else:
                    pass
                    # mspp_img = dbscanGray.dbscanClustering(img, eps, minPts)
                cv.imshow('Clustered', mspp_img[0])
                cv.waitKey(0)
                repeat = int(input('WannaRepeat? 1Yes 0No: '))
                if repeat != 0:
                    continue
                cv.destroyAllWindows()
                size_string = '320x240'
                filename = 'resultnt/TRYSLICING' + str(repeat) + color_string + size_string + image_name
                status = cv.imwrite(filename, mspp_img[0])
                print('Image written: %s - Status: %s' % (image_name, status))
                with open('resultnt/record.txt', 'a') as f:
                    exec_time = mspp_img[1]
                    result = str(repeat) + ' eps: ' + str(eps) + ' minPts: ' + str(minPts) + ' ' + image_name + ': ' + \
                             str(exec_time) + '\n'
                    f.write(result)
                    f.close()'''
    ARItotalScore = 0
    AMItotalScore = 0
    datasetPath = '../DATASET/'
    resultPath = '../DATASET/resultDBSCAN/'
    imgSavePath = join(resultPath, 'predicted/')
    gtSavePath = join(resultPath, 'true_GT/')
    groundTruthPath = join(datasetPath, 'groundTruth/train/')
    imagesPath = join(datasetPath, 'images/train/')
    imgDirFiles = [f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]
    gtDirFiles = [f for f in listdir(groundTruthPath) if isfile(join(groundTruthPath, f))]
    totImgs = len(imgDirFiles)
    for i in range(totImgs):
        # currImgName, currGTName in imgDirFiles, gtDirFiles:
        currImgName = imgDirFiles[i]
        currGTName = gtDirFiles[i]
        imgPath = join(imagesPath, currImgName)
        gtPath = join(groundTruthPath, currGTName)
        img = cv.imread(imgPath, cv.IMREAD_COLOR)
        imgGT = io.loadmat(gtPath)
        imgGT = imgGT['groundTruth'][0][0][0][0][0]
        imgGT = cv.resize(imgGT, (0, 0), fx=0.5, fy=0.5)
        # SAVING THE IMAGE GROUND TRUTH
        print(currGTName)
        currGTName = currGTName[:-4]
        print(currGTName)
        plt.imsave(gtSavePath + str(currGTName) + '.png', imgGT)
        # plt.imshow(imgGT)
        # plt.axis('off')
        # plt.savefig(gtSavePath + str(currImgName) + '.png')
        plt.close()
        gtShape = imgGT.shape
        img = cv.resize(img, (gtShape[1], gtShape[0]))
        true_labels = imgGT.ravel()
        h = 20  # range of color
        minPts = 5
        clusteredImg = dbscan.dbscanClustering(img, h, minPts)
        labeledImg = rg.labeling(clusteredImg[0], h * 3)
        pred_labels = np.array(labeledImg[1]).ravel()
        tot_clusters = labeledImg[2]
        print(true_labels.shape)
        print(pred_labels.shape)
        ARIscore = adjusted_rand_score(true_labels, pred_labels)
        AMIscore = adjusted_mutual_info_score(true_labels, pred_labels)
        ARItotalScore += ARIscore
        AMItotalScore += AMIscore
        print('ARI SCORE: ', ARIscore)
        print('AMI SCORE: ', AMIscore)

        # SAVING IMG AND INFO OF CLUSTERING
        status = cv.imwrite(imgSavePath + str(currImgName), clusteredImg[0])
        print('Image written: %s - Status: %s' % (currImgName, status))
        with open(imgSavePath + 'record.txt', 'a') as f:
            exec_time = clusteredImg[1]
            result = 'H value: ' + str(h) + ' ' + currImgName + ':\n' + str(exec_time) + '\nTotC: ' + str(
                tot_clusters) + \
                     '\nARI: ' + str(ARIscore) + '\nAMI: ' + str(AMIscore) + '\n\n'
            f.write(result)
            f.close()
    with open(imgSavePath + 'record.txt', 'a') as f:
        ARIaverageScore = ARItotalScore / totImgs
        AMIaverageScore = AMItotalScore / totImgs
        result = 'ARI AverageScore: ' + str(ARIaverageScore) + '\nAMI AverageScore: ' + str(AMIaverageScore) + '\n\n'
        f.write(result)
        f.close()


if __name__ == '__main__':
    main()
