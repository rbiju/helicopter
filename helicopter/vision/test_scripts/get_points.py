import cv2
import numpy as np


def get_points(img_path: str):
    ir_frame = cv2.imread(img_path)[:, :, 0]
    edges = cv2.Canny(ir_frame, 500, 600)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    gofs = []
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (MA, ma), angle = ellipse
            if cv2.contourArea(contour) > 0:
                ellipses.append(ellipse)
                gofs.append(np.abs(1. - ((np.pi * (MA / 2.) * (ma / 2.)) / cv2.contourArea(contour))))

    img_copy = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    for ellipse, gof in zip(ellipses, gofs):
        (cx, cy), (MA, ma), angle = ellipse
        center = (int(cx), int(cy))
        axes = (int(MA / 2), int(ma / 2))

        if gof < 0.30:
            cv2.ellipse(img_copy, center, axes, angle, 0, 360, (0, 255, 0), 1)
            # cv2.putText(img_copy, f'{gof:.3f}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_copy


def draw_subpixel_circle(center, radius, shift=4):
    factor = 1 << shift

    cx, cy = center

    cx_rounded = round(cx * factor)
    cy_rounded = round(cy * factor)
    radius_rounded = round(radius * factor)

    return (cx_rounded, cy_rounded), radius_rounded


def get_points_blobs(img_path: str):
    ir_frame = cv2.imread(img_path)[:, :, 0]

    params = cv2.SimpleBlobDetector.Params()

    params.filterByColor = True
    params.blobColor = 255

    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector.create(params)

    keypoints = detector.detect(ir_frame)

    shift = 4
    mask = np.zeros(ir_frame.shape, dtype=np.uint8)
    radius_sum = 0
    for kp in keypoints:
        center, radius = draw_subpixel_circle(kp.pt, kp.size / 2, shift)
        cv2.circle(mask, center, radius, 1, -1, shift=shift)
        radius_sum += (kp.size / 2)

    avg_radius = int(radius_sum / len(keypoints))
    gaussian_mask = cv2.GaussianBlur(mask.astype(float), (avg_radius, avg_radius),
                                     avg_radius * 0.9) * mask

    output = cv2.drawKeypoints(ir_frame, keypoints, np.array([]), (0, 255, 0),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output


def detect_points_daytime(ir_frame, depth_frame):
    ir_frame = np.load(ir_frame)
    depth_frame = np.load(depth_frame)
    valid_mask = 0 < depth_frame

    params = cv2.SimpleBlobDetector.Params()

    params.filterByArea = True
    params.minArea = 7
    params.maxArea = 200

    params.filterByColor = True
    params.blobColor = 255

    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    params.filterByCircularity = True
    params.minCircularity = 0.5

    params.filterByConvexity = True
    params.minConvexity = 0.9

    params.thresholdStep = 5
    params.minThreshold = 20
    params.maxThreshold = 255

    detector = cv2.SimpleBlobDetector.create(params)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(ir_frame, cv2.MORPH_TOPHAT, kernel)

    clahe_operator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    clahe = clahe_operator.apply(tophat)

    for _ in range(2):
        clahe = clahe_operator.apply(clahe)

    keypoints = detector.detect(clahe)

    detections = cv2.drawKeypoints(tophat, keypoints, np.array([]), (0, 255, 0),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    filtered_keypoints = []
    points = []
    shift = 4
    for kp in keypoints:
        mask = np.zeros(ir_frame.shape, dtype=np.uint8)
        center, radius = draw_subpixel_circle(kp.pt, kp.size / 2, shift)
        cv2.circle(mask, center, radius, 1, -1, shift=shift)

        ksize = int(radius / (1 << shift)) | 1
        gaussian_mask = cv2.GaussianBlur(mask.astype(float), (ksize, ksize),
                                         radius * 0.9) * valid_mask

        if gaussian_mask.sum() <= 0:
            continue

        gaussian_mask = gaussian_mask / gaussian_mask.sum()
        depth = np.sum(depth_frame * gaussian_mask)

        if depth > 1.0:
            continue

        filtered_keypoints.append(kp)

        points.append([kp.pt[0], kp.pt[1], depth])

    final_detections = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    for kp in filtered_keypoints:
        cv2.circle(final_detections, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (0, 255, 0), 1)

    return detections, final_detections


if __name__ == "__main__":
    _ir_path = '/home/ray/projects/helicopter/helicopter/vision/test_scripts/ir_frame.npy'
    _depth_path = '/home/ray/projects/helicopter/helicopter/vision/test_scripts/depth_frame.npy'
    _img = detect_points_daytime(_ir_path, _depth_path)

    print('done')
