from pathlib import Path
import os
import cv2
from imutils.perspective import four_point_transform
from torch import constant_pad_nd
import numpy as np
from scipy.spatial import distance as dist
from pylsd.lsd import lsd
import itertools
import math

# add thresh bar
# showcontours live
class Scanner:
    '''
    TODO:
    ADD docstrings
    scan when document is % of whole screen
    '''


    def open(self, filename):
        try:
            self.img = cv2.imread(filename)
        except:
            print("Can't open the file")
        

    def crop(self, img_dir = None,box= None):
        if img_dir != None:
            self.img = cv2.imread(img_dir)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny= cv2.Canny(gray, 70,120)
        # do adaptive threshold on gray image
        thresh = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 2)
        thresh = 255-thresh

        # # apply morphology
        kernel = np.ones((2,2), np.uint8)
        rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)
        
        if box is not None:
            return box,four_point_transform(self.img, box)
        # get largest contour
        contours = cv2.findContours(rect,  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            area_thresh = 0
            area = cv2.contourArea(c)
            if area > area_thresh:
                area = area_thresh
                big_contour = c
        # get rotated rectangle from contour
        rot_rect = cv2.minAreaRect(big_contour)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        
       

        return box,four_point_transform(self.img, box)
        
    def save(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpen = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # apply adaptive threshold to get black and white effect
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)   
        return thresh


def angle_range(quad):
    """
    Returns the range between max and min interior angles of quadrilateral.
    The input quadrilateral must be a numpy array with vertices ordered clockwise
    starting with the top left vertex.
    """
    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles)

def filter_corners( corners, min_dist=20):
    """Filters corners that are within min_dist of others"""
    def predicate(representatives, corner):
        return all(dist.euclidean(representative, corner) >= min_dist
                    for representative in representatives)

    filtered_corners = []
    for c in corners:
        if predicate(filtered_corners, c):
            filtered_corners.append(c)
    return filtered_corners

def angle_between_vectors_degrees(u, v):
    """Returns the angle between two vectors in degrees"""
    return np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

def get_angle(p1, p2, p3):
    """
    Returns the angle between the line segment from p2 to p1 
    and the line segment from p2 to p3 in degrees
    """
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    avec = a - b
    cvec = c - b

    return angle_between_vectors_degrees(avec, cvec)


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
     
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype = "float32")

def get_corners( img):
    """
    Returns a list of corners ((x, y) tuples) found in the input image. With proper
    pre-processing and filtering, it should output at most 10 potential corners.
    This is a utility function used by get_contours. The input image is expected 
    to be rescaled and Canny filtered prior to be passed in.
    """
    lines = lsd(img)

    # massages the output from LSD
    # LSD operates on edges. One "line" has 2 edges, and so we need to combine the edges back into lines
    # 1. separate out the lines into horizontal and vertical lines.
    # 2. Draw the horizontal lines back onto a canvas, but slightly thicker and longer.
    # 3. Run connected-components on the new canvas
    # 4. Get the bounding box for each component, and the bounding box is final line.
    # 5. The ends of each line is a corner
    # 6. Repeat for vertical lines
    # 7. Draw all the final lines onto another canvas. Where the lines overlap are also corners

    corners = []
    if lines is not None:
        # separate out the horizontal and vertical lines, and draw them back onto separate canvases
        lines = lines.squeeze().astype(np.int32).tolist()
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2, _ = line
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
            else:
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

        lines = []

        # find the horizontal lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_x = np.amin(contour[:, 0], axis=0) + 2
            max_x = np.amax(contour[:, 0], axis=0) - 2
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            lines.append((min_x, left_y, max_x, right_y))
            cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
            corners.append((min_x, left_y))
            corners.append((max_x, right_y))

        # find the vertical lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y = np.amin(contour[:, 1], axis=0) + 2
            max_y = np.amax(contour[:, 1], axis=0) - 2
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            lines.append((top_x, min_y, bottom_x, max_y))
            cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
            corners.append((top_x, min_y))
            corners.append((bottom_x, max_y))

        # find the corners
        corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
        corners += zip(corners_x, corners_y)

    # remove corners in close proximity
    corners = filter_corners(corners)
    return corners

def is_valid_contour(cnt, IM_WIDTH, IM_HEIGHT):
    """Returns True if the contour satisfies all requirements set at instantitation"""
    MIN_QUAD_AREA_RATIO=0.25
    MAX_QUAD_ANGLE_RANGE=40
    return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * MIN_QUAD_AREA_RATIO 
        and angle_range(cnt) < MAX_QUAD_ANGLE_RANGE)

def get_contour(rescaled_image):
    """
    Returns a numpy array of shape (4, 2) containing the vertices of the four corners
    of the document in the image. It considers the corners returned from get_corners()
    and uses heuristics to choose the four corners that most likely represent
    the corners of the document. If no corners were found, or the four corners represent
    a quadrilateral that is too small or convex, it returns the original four corners.
    """

    # these constants are carefully chosen
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    # dilate helps to remove potential holes between edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # find edges and mark them in the output map using the Canny algorithm
    edged = cv2.Canny(dilated, 0, CANNY)
    test_corners = get_corners(edged)

    approx_contours = []

    if len(test_corners) >= 4:
        quads = []

        for quad in itertools.combinations(test_corners, 4):
            points = np.array(quad)
            points = order_points(points)
            points = np.array([[p] for p in points], dtype = "int32")
            quads.append(points)

        # get top five quadrilaterals by area
        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
        # sort candidate quadrilaterals by their angle range, which helps remove outliers
        quads = sorted(quads, key=angle_range)

        approx = quads[0]
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)

        # for debugging: uncomment the code below to draw the corners and countour found 
        # by get_corners() and overlay it on the image

        # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
        # plt.scatter(*zip(*test_corners))
        # plt.imshow(rescaled_image)
        # plt.show()

    # also attempt to find contours directly from the edged image, which occasionally 
    # produces better results
    (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        approx = cv2.approxPolyDP(c, 80, True)
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)
            break

    # If we did not find any valid contours, just use the whole image
    if not approx_contours:
        TOP_RIGHT = (IM_WIDTH, 0)
        BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
        BOTTOM_LEFT = (0, IM_HEIGHT)
        TOP_LEFT = (0, 0)
        screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

    else:
        screenCnt = max(approx_contours, key=cv2.contourArea)
        
    return screenCnt.reshape(4, 2)
            
def main():
    path = '/Users/rostyslavmosorov/Desktop/projekty/receipt-scanner/src/images/code.png'
    s = Scanner()
    s.open(path)
    new_img = s.crop()

        

if __name__ == '__main__':
    # FileApp().run()
    main()