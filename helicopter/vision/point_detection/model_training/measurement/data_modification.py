import cv2
import os

folder_path = '/home/ray/datasets/helicopter/point_detection/master_tophat/images/val'

valid_extensions = '.png'


def process_images(directory):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))
    clahe_operator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))

    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(directory, filename)

            img = cv2.imread(file_path)

            if img is None:
                print(f"Skipping {filename}: Could not read file.")
                continue

            img = img[:, :, 0]
            processed_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            processed_img = clahe_operator.apply(processed_img)

            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(file_path, processed_img)
            print(f"Successfully processed: {filename}")


if __name__ == "__main__":
    process_images(folder_path)