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

    ellipses = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2) + 3

        if y - r < 0 or x - r < 0:
            continue
        if y + r > ir_frame.shape[0] or x + r > ir_frame.shape[1]:
            continue

        roi = ir_frame[y - r:y + r, x - r:x + r]

        threshold_value = int(np.percentile(roi, 50))
        _, thresh = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipseDirect(cnt)
                cx, cy = ellipse[0][0] + (x - r), ellipse[0][0] + (y - r)
                world_ellipse = ((cx, cy), ellipse[1], ellipse[2])
                ellipses.append(world_ellipse)

    img_copy = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    for ellipse in ellipses:
        (cx, cy), (MA, ma), angle = ellipse
        center = (int(cx), int(cy))
        axes = (int(MA / 2), int(ma / 2))
        cv2.ellipse(img_copy, center, axes, angle, 0, 360, (0, 0, 255), 1)

    output = cv2.drawKeypoints(ir_frame, keypoints, np.array([]), (0, 255, 0),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output


if __name__ == "__main__":
    _img_path = '/home/ray/projects/helicopter/helicopter/vision/test_scripts/sample.png'
    _img = get_points_blobs(_img_path)

    print('done')
