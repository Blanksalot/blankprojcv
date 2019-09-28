import numpy as np
import cv2


def applyHomography(pos1: np.array, h12: np.ndarray):
    """
    Transform coordinates pos1 to pos2 using homography H12.
    
    :param pos1: An nx2 matrix of [x,y] point coordinates per row. 
    :param h12: A 3x3 homography matrix. 
    :return: pos2: An nx2 matrix of [x,y] point coordinates per row obtained from transforming pos1 using H12.
    """
    dv = np.ones((3, 1))
    dv[1, 1] = pos1[0]
    dv[2, 1] = pos1[1]
    pres = h12.dot(dv)
    res = pres[:2, ]
    res[0, 1] /= pres[2, 1]
    res[1, 1] /= pres[2, 1]
    return res


def ransacHomography(pos1, pos2, num_iters, inlier_tol):
    """
    Fit homography to maximal inliers given point matches using the RANSAC algorithm.
    :param pos1: nx2 matrix containing n rows of [x,y] coordinates of matched points.
    :param pos2: nx2 matrix containing n rows of [x,y] coordinates of matched points.
    :param num_iters: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :return:    H12: A 3x3 normalized homography matrix
                inliers: An array containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    pass

def displayMatches(im1, im2, pos1, pos2, inlind):
    """"
    % DISPLAYMATCHES Display matched pt. pairs overlayed on given image pair.
    % Arguments:
    % im1,im2 􀀀 two grayscale images
    3
    % pos1,pos2 􀀀 nx2 matrices containing n rows of [x,y] coordinates of matched
    % points in im1 and im2 (i.e. the i'th match's coordinate is
    % pos1(i,:) in im1 and and pos2(i,:) in im2).
    % inlind 􀀀 k􀀀element array of inlier matches (e.g. see output of
    % ransacHomography)
    """
    pass
