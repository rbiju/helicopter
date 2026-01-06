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


if __name__ == "__main__":
    _img_path = '/home/ray/projects/helicopter/helicopter/vision/test_scripts/sample.png'
    _img = get_points(_img_path)

    print('done')
