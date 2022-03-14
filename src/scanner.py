from imutils.perspective import four_point_transform
import numpy as np
from scipy.spatial import distance as dist
from pylsd.lsd import lsd

import cv2
import itertools
import math


class Scanner:
    def open(self, filename):
        try:
            self.img = cv2.imread(filename)
        except:
            print("Can't open the file")

    def crop(self, contour):
        img = four_point_transform(self.img, contour)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        self.img = cv2.adaptiveThreshold(
            sharpen,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            15,
        )
        return self.img


def angle_range(quad):

    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles)


def filter_corners(corners, min_dist=20):
    def predicate(representatives, corner):
        return all(
            dist.euclidean(representative, corner) >= min_dist
            for representative in representatives
        )

    filtered_corners = []
    for c in corners:
        if predicate(filtered_corners, c):
            filtered_corners.append(c)
    return filtered_corners


def angle_between_vectors_degrees(u, v):
    return np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    )


def get_angle(p1, p2, p3):

    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    avec = a - b
    cvec = c - b

    return angle_between_vectors_degrees(avec, cvec)


def order_points(pts):

    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def get_corners(img):

    lines = lsd(img)

    corners = []
    if lines is not None:

        lines = lines.squeeze().astype(np.int32).tolist()
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2, _ = line
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(
                    ((x1, y1), (x2, y2)), key=lambda pt: pt[0]
                )
                cv2.line(
                    horizontal_lines_canvas,
                    (max(x1 - 5, 0), y1),
                    (min(x2 + 5, img.shape[1] - 1), y2),
                    255,
                    2,
                )
            else:
                (x1, y1), (x2, y2) = sorted(
                    ((x1, y1), (x2, y2)), key=lambda pt: pt[1]
                )
                cv2.line(
                    vertical_lines_canvas,
                    (x1, max(y1 - 5, 0)),
                    (x2, min(y2 + 5, img.shape[0] - 1)),
                    255,
                    2,
                )

        lines = []

        (contours, hierarchy) = cv2.findContours(
            horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(
            contours, key=lambda c: cv2.arcLength(c, True), reverse=True
        )[:2]
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_x = np.amin(contour[:, 0], axis=0) + 2
            max_x = np.amax(contour[:, 0], axis=0) - 2
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            lines.append((min_x, left_y, max_x, right_y))
            cv2.line(
                horizontal_lines_canvas,
                (min_x, left_y),
                (max_x, right_y),
                1,
                1,
            )
            corners.append((min_x, left_y))
            corners.append((max_x, right_y))

        (contours, hierarchy) = cv2.findContours(
            vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(
            contours, key=lambda c: cv2.arcLength(c, True), reverse=True
        )[:2]
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y = np.amin(contour[:, 1], axis=0) + 2
            max_y = np.amax(contour[:, 1], axis=0) - 2
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            lines.append((top_x, min_y, bottom_x, max_y))
            cv2.line(
                vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1
            )
            corners.append((top_x, min_y))
            corners.append((bottom_x, max_y))

        corners_y, corners_x = np.where(
            horizontal_lines_canvas + vertical_lines_canvas == 2
        )
        corners += zip(corners_x, corners_y)

    corners = filter_corners(corners)
    return corners


def is_valid_contour(cnt, IM_WIDTH, IM_HEIGHT):
    MIN_QUAD_AREA_RATIO = 0.25
    MAX_QUAD_ANGLE_RANGE = 40
    return (
        len(cnt) == 4
        and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * MIN_QUAD_AREA_RATIO
        and angle_range(cnt) < MAX_QUAD_ANGLE_RANGE
    )


def get_contour(rescaled_image):
    MORPH = 9
    CANNY = 84

    IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

    gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    edged = cv2.Canny(dilated, 0, CANNY)
    test_corners = get_corners(edged)

    approx_contours = []

    if len(test_corners) >= 4:
        quads = []

        for quad in itertools.combinations(test_corners, 4):
            points = np.array(quad)
            points = order_points(points)
            points = np.array([[p] for p in points], dtype="int32")
            quads.append(points)

        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]

        quads = sorted(quads, key=angle_range)

        approx = quads[0]
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)

    (cnts, hierarchy) = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:

        approx = cv2.approxPolyDP(c, 80, True)
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)
            break

    if not approx_contours:
        TOP_RIGHT = (IM_WIDTH, 0)
        BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
        BOTTOM_LEFT = (0, IM_HEIGHT)
        TOP_LEFT = (0, 0)
        screenCnt = np.array(
            [[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]]
        )

    else:
        screenCnt = max(approx_contours, key=cv2.contourArea)

    return screenCnt.reshape(4, 2)


def main():
    path = "/Users/rostyslavmosorov/Desktop/projekty/receipt-scanner/src/images/code.png"
    s = Scanner()
    s.open(path)


if __name__ == "__main__":

    main()
