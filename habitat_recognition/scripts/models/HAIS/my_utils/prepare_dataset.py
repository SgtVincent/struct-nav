##### This script is to rearrange the files in ScanNet dataset according to the requirement of this repository
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_dir", type=str, default="dataset/scannetv2/meta_data"
    )
    parser.add_argument(
        "--src_scans_dir",
        type=str,
        default="/media/junting/SSD_data/ScanNet/scans",
    )
    parser.add_argument(
        "--src_scans_test_dir",
        type=str,
        default="/media/junting/SSD_data/ScanNet/scans_test",
    )
    parser.add_argument(
        "--dst_dataset_dir",
        type=str,
        default="/home/junting/project_cvl/HAIS/dataset/scannetv2",
    )
    parser.add_argument(
        "--mode", type=str, default="soft_link", choices=["soft_link", "copy"]
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.meta_dir, "scannetv2_train.txt"), "r") as f:
        train_scenes = f.read().splitlines()
    with open(os.path.join(args.meta_dir, "scannetv2_val.txt"), "r") as f:
        val_scenes = f.read().splitlines()
    with open(os.path.join(args.meta_dir, "scannetv2_test.txt"), "r") as f:
        test_scenes = f.read().splitlines()
    train_dataset_dir = os.path.join(args.dst_dataset_dir, "train")
    val_dataset_dir = os.path.join(args.dst_dataset_dir, "val")
    test_dataset_dir = os.path.join(args.dst_dataset_dir, "test")

    # train
    # [scene_id]_vh_clean_2.ply, [scene_id]_vh_clean_2.labels.ply,
    # [scene_id]_vh_clean_2.0.010000.segs.json,[scene_id].aggregation.json
    if args.mode == "soft_link":
        for scene in train_scenes:
            scene_dir = os.path.join(args.src_scans_dir, scene)
            os.symlink(
                os.path.join(scene_dir, f"{scene}_vh_clean_2.ply"),
                os.path.join(train_dataset_dir, f"{scene}_vh_clean_2.ply"),
            )
            os.symlink(
                os.path.join(scene_dir, f"{scene}_vh_clean_2.labels.ply"),
                os.path.join(
                    train_dataset_dir, f"{scene}_vh_clean_2.labels.ply"
                ),
            )
            os.symlink(
                os.path.join(
                    scene_dir, f"{scene}_vh_clean_2.0.010000.segs.json"
                ),
                os.path.join(
                    train_dataset_dir, f"{scene}_vh_clean_2.0.010000.segs.json"
                ),
            )
            os.symlink(
                os.path.join(scene_dir, f"{scene}.aggregation.json"),
                os.path.join(train_dataset_dir, f"{scene}.aggregation.json"),
            )

        # val
        # [scene_id]_vh_clean_2.ply, [scene_id]_vh_clean_2.labels.ply,
        # [scene_id]_vh_clean_2.0.010000.segs.json,[scene_id].aggregation.json
        for scene in val_scenes:
            scene_dir = os.path.join(args.src_scans_dir, scene)
            os.symlink(
                os.path.join(scene_dir, f"{scene}_vh_clean_2.ply"),
                os.path.join(val_dataset_dir, f"{scene}_vh_clean_2.ply"),
            )
            os.symlink(
                os.path.join(scene_dir, f"{scene}_vh_clean_2.labels.ply"),
                os.path.join(
                    val_dataset_dir, f"{scene}_vh_clean_2.labels.ply"
                ),
            )
            os.symlink(
                os.path.join(
                    scene_dir, f"{scene}_vh_clean_2.0.010000.segs.json"
                ),
                os.path.join(
                    val_dataset_dir, f"{scene}_vh_clean_2.0.010000.segs.json"
                ),
            )
            os.symlink(
                os.path.join(scene_dir, f"{scene}.aggregation.json"),
                os.path.join(val_dataset_dir, f"{scene}.aggregation.json"),
            )

        # test
        # [scene_id]_vh_clean_2.ply
        for scene in test_scenes:
            scene_dir = os.path.join(args.src_scans_test_dir, scene)
            os.symlink(
                os.path.join(scene_dir, f"{scene}_vh_clean_2.ply"),
                os.path.join(test_dataset_dir, f"{scene}_vh_clean_2.ply"),
            )
    elif args.mode == "copy":
        raise NotImplementedError
    else:
        raise NotImplementedError
