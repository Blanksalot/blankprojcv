import math
import numpy as np
from numpy.linalg import inv
import cv2
from cv2 import *
import random
from matplotlib import pyplot as plt
import glob
import sys
from imutils import paths

def read_frames(path):
    data = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rgb = np.fliplr(data.reshape(-1, 3)).reshape(data.shape)
    return rgb


def match_feature_points(img1, img2):
    def findPoints(img):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

        return kp, des

    def matchFeatures(des1, des2):
        topmatches = list()
        idx1 = 0
        for i in des1:
            allDis = list()
            idx2 = 0
            for j in des2:
                d = cv2.norm(i, j)
                item = [d, idx1, idx2]
                idx2 += 1
                allDis.append(item)
            idx1 += 1
            allDis.sort()
            topmatches.append(allDis[0:2])

        return topmatches

    def goodMatches(matches):
        good = []
        for m, n in matches:
            # print("m[0]= ", m[0], " ,n[0]= ", n[0])
            if m[0] < 0.5 * n[0]:
                good.append(m)

        return good

    kp1, des1 = findPoints(img1)
    kp2, des2 = findPoints(img2)
    print("keypoint detected")
    """............Matching keypoints in both images................"""

    matches = matchFeatures(des1, des2)
    print("matching done")
    """........Finding good matches.............."""
    good = goodMatches(matches)
    print("Length of good = ", len(good))
    # print("Good = ", good)

    MIN_MATCH_COUNT = 30
    if len(good) > MIN_MATCH_COUNT:
        coordList = list()
        pos1 = []
        pos2 = []
        for m in good:
            (x1, y1) = kp1[m[1]].pt
            pos1.append([x1, y1])
            (x2, y2) = kp2[m[2]].pt
            pos2.append([x2, y2])
            coordList.append([x1, y1, x2, y2])
        return pos1, pos2


    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


def applyHomography(pos1: np.array, h12: np.ndarray):
    """
    Transform coordinates pos1 to pos2 using homography H12.
    
    :param pos1: An nx2 matrix of [x,y] point coordinates per row. 
    :param h12: A 3x3 homography matrix. 
    :return: pos2: An nx2 matrix of [x,y] point coordinates per row obtained from transforming pos1 using H12.
    """
    dv = np.ones((3, 1))
    res = np.zeros(pos1.shape)
    for i in range(pos1.shape[0]):
        dv[1] = pos1[i, 0]
        dv[2] = pos1[i, 1]
        pres = h12.dot(dv)

        res[i, 0] = pres[0] / pres[2]
        res[i, 1] /= pres[1] / pres[2]
    return res





def ransacHomography(pos1, pos2, num_iters=200, inlier_tol=5):
    """
    Fit homography to maximal inliers given point matches using the RANSAC algorithm.
    :param pos1: nx2 matrix containing n rows of [x,y] coordinates of matched points.
    :param pos2: nx2 matrix containing n rows of [x,y] coordinates of matched points.
    :param num_iters: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :return:    H12: A 3x3 normalized homography matrix
                inliers: An array containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    def find_homography(random_4_pts):
        homoList = list()
        for pt in random_4_pts:
            # print("pt.item(0) ", pt.item(0))
            xVal = [-pt.item(0), -pt.item(1), -1, 0, 0, 0, pt.item(2) * pt.item(0), pt.item(2) * pt.item(1), pt.item(2)]
            yVal = [0, 0, 0, -pt.item(0), -pt.item(1), -1, pt.item(3) * pt.item(0), pt.item(3) * pt.item(1), pt.item(3)]
            homoList.append(xVal)
            homoList.append(yVal)

        homoMat = np.matrix(homoList)

        u, s, v = np.linalg.svd(homoMat)

        h = np.reshape(v[8], (3, 3))
        h = (1 / h.item(8)) * h

        return h

    def calculate_distance(i, homo):
        p1 = np.transpose(np.matrix([i[0].item(0), i[0].item(1), 1]))
        estimatep2 = np.dot(homo, p1)
        estimatep2 = (1 / estimatep2.item(2)) * estimatep2

        p2 = np.transpose(np.matrix([i[0].item(2), i[0].item(3), 1]))
        error = p2 - estimatep2
        return np.linalg.norm(error)

    def ransac_imp(cmat):
        maxInlier = []
        H = None
        for j in range(num_iters):
            randFour = []
            for i in range(0, 4):
                randmatch = random.choice(cmat)
                randFour.append(randmatch)
            homo = find_homography(randFour)
            # print("Homography function output: ", homo)
            inlier = list()
            for i in cmat:
                dist = calculate_distance(i, homo)

                if dist < inlier_tol:
                    inlier.append(i)

            if len(inlier) > len(maxInlier):
                maxInlier = inlier
                H = homo
        if not H is None:
            raise RuntimeError('Unable to generate H via RANSAC')
        return maxInlier, H

    coor_mat = []
    for i in range(len(pos1)):
        coor_mat.append([pos1[i, 0], pos1[i, 1], pos2[i, 0], pos2[i, 1]])
    coord_matrix = np.matrix(coor_mat)
    H, mask = cv2.findHomography(pos1, pos2, 8, maxIters=num_iters)
    inliers = []
    for i in range(len(pos1)):
        if mask[i, 0] == 1:
            inliers.append([pos1[i, 0], pos1[i, 1], pos2[i, 0], pos2[i, 1]])
    inls = np.matrix(inliers)
    print(len(pos1))
    print(len(inliers))
    return inls, H
    return ransac_imp(coord_matrix)






def displayMatches(im1, im2, pos1: np.ndarray, pos2, inlind):
    """
    Display matched pt. pairs overlayed on given image pair.
    :param im1: grayscale image
    :param im2: grayscale image
    :param pos1: nx2 matrix containing n rows of [x,y] coordinates of matched points in im1 and im2 (i.e. the i'th match's coordinate is pos1(i,:) in im1 and and pos2(i,:) in im2).
    :param pos2: nx2 matrix containing n rows of [x,y] coordinates of matched points in im1 and im2 (i.e. the i'th match's coordinate is pos1(i,:) in im1 and and pos2(i,:) in im2).
    :param inlind: k element array of inlier matches (e.g. see output of ransacHomography)
    """
    combo = np.hstack((im1, im2))
    plt.imshow(combo, 'gray')
    p1x, p1y = np.hsplit(pos1, 2)
    p2x, p2y = np.hsplit(pos2, 2)
    p2x = [mem + im1.shape[1] for mem in p2x]

    for i in range(len(p1x)):
        plt.plot([p1x[i], p2x[i]], [p1y[i], p2y[i]], 'b-')
    plt.scatter(p1x, p1y, c='red', marker='.')
    plt.scatter(p2x, p2y, c='red', marker='.')
    if inlind is not None:
        for i in range(len(inlind)):
            plt.plot([inlind[i][0, 0], inlind[i][0, 2] + im1.shape[1]], [inlind[i][0, 1], inlind[i][0, 3]], 'y-')
    plt.show()


def accumulateHomographies(h_pair, m):
    """
    Accumulate homography matrix sequence.
    :param h_pair: array or dict of M􀀀1 3x3 homography matrices where Hpairfig is a homography that transforms between coordinate systems i and i+1.
    :param m: Index of coordinate system we would like to accumulate the given homographies towards (see details below)
    :return: h_tot: array or dict of M 3x3 homography matrices where Htotfig transforms coordinate system i to the coordinate system having the index m
    """
    # m = math.ceil(len(h_pair)/2)
    htot = [np.ones((3, 3))] * (len(h_pair) + 1)
    for i in range(len(h_pair) + 1):
        if i == m:
            htot[i] = np.eye(3)
        elif i < m:
            for ind in range(i, m):
                htot[i] = htot[i].dot(h_pair[ind])
        elif i > m:
            for ind in range(m, i):
                    htot[i] = htot[i].dot(inv(h_pair[ind]))

    return htot

def make_pano_mat(im, H):
    y_max = x_max = -1 * (sys.maxsize - 1)
    y_min = x_min = sys.maxsize
    for i in range(len(im)):
        old_pts = np.ndarray(shape=(4, 2))
        old_pts[0, 0] = 0
        old_pts[0, 1] = 0
        old_pts[1, 0] = im[i].shape[0]
        old_pts[1, 1] = 0
        old_pts[2, 0] = 0
        old_pts[2, 1] = im[i].shape[1]
        old_pts[3, 0] = im[i].shape[0]
        old_pts[3, 1] = im[i].shape[1]

        new_pts = applyHomography(old_pts, H[i])
        for newp in new_pts:
            x_max = newp[0] if newp[0] > x_max else x_max
            y_max = newp[1] if newp[1] > y_max else y_max

            x_min = newp[0] if newp[0] < x_min else x_min
            y_min = newp[1] if newp[1] < y_min else y_min

    return np.zeros((int(x_max-x_min), int(y_max-y_min)))

def split_into_sections(im):
    borders = list()
    prev = 0
    for i in range(len(im)-1):
        border = prev + (im[i].shape[1]/2 + im[i].shape[1]/2)/2
        prev = border
        borders.append(border)

    return borders

def renderPanorama(im, H):
    """
    Renders a set of images into a combined panorama image.
    :param im: array or dict of n grayscale images
    :param H: array or dict array of n 3x3 homography matrices transforming the ith image coordinates to the panorama image coordinates.
    :return: panorama: A grayscale panorama image composed of n vertical strips that were backwarped each from the relevant frame imfig using homography Hfig
    """
    pmat = make_pano_mat(im, H)
    sections = split_into_sections(im)
    for i, img in enumerate(im):
        low = 0 if i == 0 else sections[i-1]
        high = sections[i]
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                point = np.ndarray((2, 1))
                point[0, 0] = j
                point[1, 0] = k
                coord = applyHomography(point, H[i])
                if low < coord[0, 0] <= high:
                    pmat[coord[0, 0], coord[1, 0]] = img[j, k]

    return pmat


def pano_all(Hs, data, out):
    try:
        pano = list()
        Rs = []
        Gs = []
        Bs = []

        for d in data:
            r, g, b = cv2.split(d)
            Rs.append(r)
            Gs.append(g)
            Bs.append(b)
        pano.append(renderPanorama(Rs, Hs))
        pano.append(renderPanorama(Gs, Hs))
        pano.append(renderPanorama(Bs, Hs))

        res = cv2.merge(pano)
    except Exception:
        st = cv2.createStitcher()
        status, res = st.stitch(data)
        if status !=0:
            res = None
    cv2.imwrite(out, res)


def generatePanorama(path=None, out='out.jpg'):
    if not path:
        path = r'C:\workspace\blankprojcv\data\inp\examples\ox\*.jpg'
    else:
        path += r'\*.jpg'
    im_paths = glob.glob(path)
    if len(im_paths) < 2:
        raise RuntimeError('need at least 2 pictures')
    data = []
    for p in im_paths:
        im = read_frames(p)
        data.append(im)
        show(im)

    rH = []
    for i in range(len(im_paths)-1):
        img1 = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(data[i+1], cv2.COLOR_BGR2GRAY)
        pos1, pos2 = match_feature_points(img1, img2)
        pos1 = np.vstack(pos1)
        pos2 = np.vstack(pos2)
        maxInlier, H = ransacHomography(pos1, pos2)
        rH.append(H)
        displayMatches(img1, img2, pos1, pos2, maxInlier)
    Hs = accumulateHomographies(rH, math.ceil(len(data)/2))
    pano_all(Hs, data, out)







def show(data):
    if len(data.shape) == 3:
        plt.imshow(cv2.cvtColor(data, cv2.COLOR_RGB2BGR), cmap='gray', interpolation='bicubic')
    else:
        plt.imshow(data, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == '__main__':
    generatePanorama()