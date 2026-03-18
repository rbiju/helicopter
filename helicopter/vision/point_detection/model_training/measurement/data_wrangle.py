import shutil
import random
from pathlib import Path


def merge_yolo_datasets(source_dirs, output_dir, split_ratio=0.8):
    output_path = Path(output_dir)

    dirs = {
        'train_img': output_path / 'images' / 'train',
        'val_img': output_path / 'images' / 'val',
        'train_lbl': output_path / 'labels' / 'train',
        'val_lbl': output_path / 'labels' / 'val'
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    valid_exts = {'.png'}

    total_files = 0

    print(f"Processing {len(source_dirs)} directories...")

    for source in source_dirs:
        source_path = Path(source)

        dataset_prefix = source_path.name

        img_source = source_path / 'images'
        lbl_source = source_path / 'labels'

        if not img_source.exists() or not lbl_source.exists():
            print(f"Skipping {source}: Missing 'images' or 'labels' subdirectory.")
            continue

        # Gather all images
        images = [f for f in img_source.iterdir() if f.suffix.lower() in valid_exts]

        for img_file in images:
            # Identify corresponding label file
            label_file = lbl_source / f"{img_file.stem}.txt"

            if not label_file.exists():
                print(f"Warning: Label not found for {img_file.name}, skipping.")
                continue

            is_train = random.random() < split_ratio

            new_stem = f"{dataset_prefix}_{img_file.stem}"
            new_img_name = f"{new_stem}{img_file.suffix}"
            new_lbl_name = f"{new_stem}.txt"

            dest_img_key = 'train_img' if is_train else 'val_img'
            dest_lbl_key = 'train_lbl' if is_train else 'val_lbl'

            shutil.copy2(img_file, dirs[dest_img_key] / new_img_name)
            shutil.copy2(label_file, dirs[dest_lbl_key] / new_lbl_name)

            total_files += 1

    print(f"Done! Merged {total_files} pairs into '{output_dir}'.")


if __name__ == "__main__":
    inputs = [
        "/home/ray/datasets/helicopter/point_detection/measure/night",
        "/home/ray/datasets/helicopter/point_detection/measure/day",
        "/home/ray/datasets/helicopter/point_detection/measure/night_noise",
        "/home/ray/datasets/helicopter/point_detection/measure/day2",
        "/home/ray/datasets/helicopter/point_detection/measure/day3",
        "/home/ray/datasets/helicopter/point_detection/measure/day4",
        "/home/ray/datasets/helicopter/point_detection/measure/day5",
        "/home/ray/datasets/helicopter/point_detection/measure/day6",
        "/home/ray/datasets/helicopter/point_detection/measure/day7",
        "/home/ray/datasets/helicopter/point_detection/measure/day8",
        "/home/ray/datasets/helicopter/point_detection/measure/day9"
    ]

    output = "/home/ray/datasets/helicopter/point_detection/measure/master"

    merge_yolo_datasets(inputs, output, split_ratio=0.9)
