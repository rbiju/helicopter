import shutil
import random
from pathlib import Path


def organize_labels(_export_path: str):
    source_path = Path(_export_path) / 'labels'
    target_path = Path(_export_path).parent

    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist.")
        return

    moved_count = 0

    for label_file in source_path.glob("*.txt"):
        filename = label_file.name

        if filename == "classes.txt":
            print("Skipping 'classes.txt'...")
            continue

        prefix = filename.split('_')[0]

        destination_dir = target_path / prefix / "labels"
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_file = destination_dir / filename

        shutil.move(label_file, destination_file)
        moved_count += 1
    print(f"\nSuccess! Moved {moved_count} label files into their respective folders.")


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
        img_source = source_path / 'images'
        lbl_source = source_path / 'labels'

        if not img_source.exists() or not lbl_source.exists():
            print(f"Skipping {source}: Missing 'images' or 'labels' subdirectory.")
            continue

        valid_images = [
            f for f in img_source.iterdir()
            if f.suffix.lower() in valid_exts and (lbl_source / f"{f.stem}.txt").exists()
        ]

        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * split_ratio)

        for i, img_file in enumerate(valid_images):
            dest_type = 'train' if i < split_idx else 'val'

            shutil.copy2(img_file, dirs[f'{dest_type}_img'] / img_file.name)
            shutil.copy2(lbl_source / f"{img_file.stem}.txt", dirs[f'{dest_type}_lbl'] / f"{img_file.stem}.txt")

            total_files += 1

    print(f"Done! Merged {total_files} pairs into '{output_dir}'.")


if __name__ == "__main__":
    inputs = [
        "/home/ray/datasets/helicopter/point_detection/tracking/set01",
        "/home/ray/datasets/helicopter/point_detection/tracking/set02",
        "/home/ray/datasets/helicopter/point_detection/tracking/set03",
        "/home/ray/datasets/helicopter/point_detection/tracking/set04",
        "/home/ray/datasets/helicopter/point_detection/tracking/set05",
        "/home/ray/datasets/helicopter/point_detection/tracking/set06",
        "/home/ray/datasets/helicopter/point_detection/tracking/set07",
        "/home/ray/datasets/helicopter/point_detection/tracking/set08",
        "/home/ray/datasets/helicopter/point_detection/tracking/set09",
        "/home/ray/datasets/helicopter/point_detection/tracking/set10",
        "/home/ray/datasets/helicopter/point_detection/tracking/set11",
        "/home/ray/datasets/helicopter/point_detection/tracking/set12",
        "/home/ray/datasets/helicopter/point_detection/tracking/set13",
        "/home/ray/datasets/helicopter/point_detection/tracking/set14",
        "/home/ray/datasets/helicopter/point_detection/tracking/set15",
        "/home/ray/datasets/helicopter/point_detection/tracking/set16",
        "/home/ray/datasets/helicopter/point_detection/tracking/set17",
        "/home/ray/datasets/helicopter/point_detection/tracking/set18",
        "/home/ray/datasets/helicopter/point_detection/tracking/set19",
        "/home/ray/datasets/helicopter/point_detection/tracking/set20",
        "/home/ray/datasets/helicopter/point_detection/tracking/set21",
        "/home/ray/datasets/helicopter/point_detection/tracking/set22",
        "/home/ray/datasets/helicopter/point_detection/tracking/set23",
        "/home/ray/datasets/helicopter/point_detection/tracking/set24",
        "/home/ray/datasets/helicopter/point_detection/tracking/set25",
        "/home/ray/datasets/helicopter/point_detection/tracking/set26",
        "/home/ray/datasets/helicopter/point_detection/tracking/set27",
    ]

    output = "/home/ray/datasets/helicopter/point_detection/tracking/master"

    # export_path = "/home/ray/datasets/helicopter/point_detection/tracking/temp"
    # organize_labels(export_path)
    merge_yolo_datasets(inputs, output, split_ratio=0.9)