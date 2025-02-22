if __name__ == "__main__" and __package__ is None:
    from os import sys

    sys.path.append("../")
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
from sg_nav_utils import dataLoaderScanNet, define, util, util_label
from utils.util_search import SAMPLE_METHODS, find_neighbors

debug = True
debug = False

name_same_segment = define.NAME_SAME_PART


def Parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Process some integers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="allow multiple rel pred outputs per pair",
        required=False,
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="ScanNet20",
        choices=["NYU40", "ScanNet20"],
        help="label",
        required=False,
    )
    parser.add_argument(
        "--pth_out",
        type=str,
        default="../data/tmp",
        help="pth to output directory",
        required=False,
    )
    parser.add_argument("--target_scan", type=str, default="", help="")

    ## options
    parser.add_argument(
        "--verbose", type=bool, default=False, help="verbal", required=False
    )
    parser.add_argument(
        "--debug", type=int, default=0, help="debug", required=False
    )

    parser.add_argument(
        "--scan_name",
        type=str,
        choices=["inseg.ply", "cvvseg.ply"],
        default="inseg.ply",
        help="",
    )

    ## neighbor search parameters
    parser.add_argument(
        "--search_method",
        type=str,
        choices=["BBOX", "KNN"],
        default="BBOX",
        help="How to split the scene.",
    )
    parser.add_argument(
        "--radius_receptive",
        type=float,
        default=0.5,
        help="The receptive field of each seed.",
    )

    # # Correspondence Parameters
    parser.add_argument(
        "--max_dist",
        type=float,
        default=0.1,
        help="maximum distance to find corresopndence.",
    )
    parser.add_argument(
        "--min_seg_size",
        type=int,
        default=512,
        help="Minimum number of points of a segment.",
    )
    parser.add_argument(
        "--corr_thres",
        type=float,
        default=0.5,
        help="How the percentage of the points to the same target segment must exceeds this value.",
    )
    parser.add_argument(
        "--occ_thres",
        type=float,
        default=0.75,
        help="2nd/1st must smaller than this.",
    )

    return parser


def load_inseg(pth_ply):
    # if pth_ply.find('inseg.ply') >=0:
    cloud_pd = trimesh.load(pth_ply, process=False)
    points_pd = cloud_pd.vertices
    segments_pd = cloud_pd.metadata["ply_raw"]["vertex"]["data"][
        "label"
    ].flatten()
    # elif pth_ply.find('cvvseg.ply') >=0:

    return cloud_pd, points_pd, segments_pd


def process(pth_scan, scan_id, label_type, verbose=False) -> list:
    # some params
    max_distance = args.max_dist
    filter_segment_size = (
        args.min_seg_size
    )  # if the num of points within a segment below this threshold, discard this
    filter_corr_thres = (
        args.corr_thres
    )  # if percentage of the corresponding label must exceed this value to accept the correspondence
    filter_occ_ratio = args.occ_thres

    pth_pd = os.path.join(define.SCANNET_DATA_PATH, scan_id, args.scan_name)
    pth_ply = os.path.join(
        define.SCANNET_DATA_PATH, scan_id, scan_id + define.SCANNET_PLY_SUBFIX
    )
    pth_agg = os.path.join(
        define.SCANNET_DATA_PATH,
        scan_id,
        scan_id + define.SCANNET_AGGRE_SUBFIX,
    )
    pth_seg = os.path.join(
        define.SCANNET_DATA_PATH, scan_id, scan_id + define.SCANNET_SEG_SUBFIX
    )

    (
        cloud_gt,
        points_gt,
        labels_gt,
        segments_gt,
    ) = dataLoaderScanNet.load_scannet(pth_ply, pth_agg, pth_seg)
    cloud_pd, points_pd, segments_pd = load_inseg(pth_pd)

    # get num of segments
    segment_ids = np.unique(segments_pd)
    segment_ids = segment_ids[segment_ids != 0]

    segs_neighbors = find_neighbors(
        points_pd,
        segments_pd,
        search_method,
        receptive_field=args.radius_receptive,
    )

    """ Check GT segments and labels """
    instance2labelName = dict()
    size_segments_gt = dict()
    uni_seg_gt_ids = np.unique(segments_gt).tolist()
    for seg_id in uni_seg_gt_ids:
        indices = np.where(segments_gt == seg_id)
        seg = segments_gt[indices]
        labels = labels_gt[indices]
        uq_label = np.unique(labels).tolist()

        if len(uq_label) > 1:
            if verbose or debug:
                print(
                    "segment",
                    seg_id,
                    "has multiple labels (",
                    uq_label,
                    ") in GT. Try to remove other labels.",
                )
            max_id = 0
            max_value = 0
            for id in uq_label:
                if verbose or debug:
                    print(
                        id,
                        len(labels[labels == id]),
                        "{:1.3f}".format(
                            len(labels[labels == id]) / len(labels)
                        ),
                    )
                if len(labels[labels == id]) > max_value:
                    max_value = len(labels[labels == id])
                    max_id = id

            for label in uq_label:
                if label == max_id:
                    continue
                if (
                    len(labels[labels == id]) > filter_segment_size
                ):  # try to generate new segment
                    new_seg_idx = max(uni_seg_gt_ids) + 1
                    uni_seg_gt_ids.append(new_seg_idx)
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = new_seg_idx
                else:
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = 0
                            labels_gt[idx] = 0  # set other label to 0
            seg = segments_gt[indices]
            labels = labels_gt[indices]
            uq_label = [max_id]

        if uq_label[0] == 0 or uq_label[0] > 40:
            name = "none"
        else:
            name = util_label.NYU40_Label_Names[uq_label[0] - 1]

        if label_type == "ScanNet20":
            if name not in util_label.SCANNET20_Label_Names:
                name = "none"

        size_segments_gt[seg_id] = len(seg)
        instance2labelName[seg_id] = name

    """ Save as ply """
    if debug:
        colors = util_label.get_NYU40_color_palette()
        cloud_gt.visual.vertex_colors = [0, 0, 0, 255]
        for seg, label_name in instance2labelName.items():
            segment_indices = np.where(segments_gt == seg)[0]
            if label_name == "none":
                continue
            label = util_label.NYU40_Label_Names.index(label_name) + 1
            for index in segment_indices:
                cloud_gt.visual.vertex_colors[index][:3] = colors[label]
        cloud_gt.export("tmp_gtcloud.ply")

    size_segments_pd = dict()

    """ Find and count all corresponding segments"""
    tree = o3d.geometry.KDTreeFlann(points_gt.transpose())
    count_seg_pd_2_corresponding_seg_gts = (
        dict()
    )  # counts each segment_pd to its corresonding segment_gt

    for segment_id in segment_ids:
        segment_indices = np.where(segments_pd == segment_id)[0]
        segment_points = points_pd[segment_indices]

        size_segments_pd[segment_id] = len(segment_points)

        if filter_segment_size > 0:
            if size_segments_pd[segment_id] < filter_segment_size:
                continue

        for i in range(len(segment_points)):
            point = segment_points[i]
            k, idx, distance = tree.search_knn_vector_3d(point, 1)
            if distance[0] > max_distance:
                continue
            segment_gt = segments_gt[idx][0]

            if segment_gt not in instance2labelName:
                continue
            if instance2labelName[segment_gt] == "none":
                continue

            if segment_id not in count_seg_pd_2_corresponding_seg_gts:
                count_seg_pd_2_corresponding_seg_gts[segment_id] = dict()
            if (
                segment_gt
                not in count_seg_pd_2_corresponding_seg_gts[segment_id]
            ):
                count_seg_pd_2_corresponding_seg_gts[segment_id][
                    segment_gt
                ] = 0
            count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] += 1

    if verbose or debug:
        print(
            "There are {} out of {} segments have found their correponding GT segments.".format(
                len(count_seg_pd_2_corresponding_seg_gts), len(segment_ids)
            )
        )
        for k, i in count_seg_pd_2_corresponding_seg_gts.items():
            print("\t{}: {}".format(k, len(i)))

    """ Find best corresponding segment """
    map_segment_pd_2_gt = dict()  # map segment_pd to segment_gt
    gt_segments_2_pd_segments = (
        dict()
    )  # how many segment_pd corresponding to this segment_gt
    for (
        segment_id,
        cor_counter,
    ) in count_seg_pd_2_corresponding_seg_gts.items():
        size_pd = size_segments_pd[segment_id]
        if verbose or debug:
            print("segment_id", segment_id, size_pd)

        max_corr_ratio = -1
        max_corr_seg = -1
        list_corr_ratio = list()
        for segment_gt, count in cor_counter.items():
            size_gt = size_segments_gt[segment_gt]
            corr_ratio = count / size_pd
            list_corr_ratio.append(corr_ratio)
            if corr_ratio > max_corr_ratio:
                max_corr_ratio = corr_ratio
                max_corr_seg = segment_gt
            if verbose or debug:
                print(
                    "\t{0:s} {1:3d} {2:8d} {3:2.3f} {4:2.3f}".format(
                        instance2labelName[segment_gt],
                        segment_gt,
                        count,
                        count / size_gt,
                        corr_ratio,
                    )
                )
        if len(list_corr_ratio) > 2:
            list_corr_ratio = sorted(list_corr_ratio, reverse=True)
            occ_ratio = list_corr_ratio[1] / list_corr_ratio[0]
        else:
            occ_ratio = 0

        if max_corr_ratio > filter_corr_thres and occ_ratio < filter_occ_ratio:
            """
            This is to prevent a segment is almost equally occupied two or more gt segments.
            """
            if verbose or debug:
                print(
                    "add correspondence of segment {:s} {:4d} to label {:4d} with the ratio {:2.3f} {:1.3f}".format(
                        instance2labelName[segment_gt],
                        segment_id,
                        max_corr_seg,
                        max_corr_ratio,
                        occ_ratio,
                    )
                )
            map_segment_pd_2_gt[segment_id] = max_corr_seg
            if max_corr_seg not in gt_segments_2_pd_segments:
                gt_segments_2_pd_segments[max_corr_seg] = list()
            gt_segments_2_pd_segments[max_corr_seg].append(segment_id)
        else:
            if verbose or debug:
                print(
                    "filter correspondence segment {:s} {:4d} to label {:4d} with the ratio {:2.3f} {:1.3f}".format(
                        instance2labelName[segment_gt],
                        segment_id,
                        max_corr_seg,
                        max_corr_ratio,
                        occ_ratio,
                    )
                )
    if verbose:
        print("final correspondence:")
        print("  pd  gt")
        for segment, label in sorted(map_segment_pd_2_gt.items()):
            print("{:4d} {:4d}".format(segment, label))
        print("final pd segments within the same gt segment")
        for gt_segment, pd_segments in sorted(
            gt_segments_2_pd_segments.items()
        ):
            print("{:4d}:".format(gt_segment), end="")
            for pd_segment in pd_segments:
                print("{:4d}".format(pd_segment), end="")
            print("")

    """ Save as ply """
    if debug:
        colors = util_label.get_NYU40_color_palette()
        cloud_pd.visual.vertex_colors = [0, 0, 0, 255]
        for segment_pd, segment_gt in map_segment_pd_2_gt.items():
            segment_indices = np.where(segments_pd == segment_pd)[0]
            label = (
                util_label.NYU40_Label_Names.index(
                    instance2labelName[segment_gt]
                )
                + 1
            )
            color = colors[label]
            for index in segment_indices:
                cloud_pd.visual.vertex_colors[index][:3] = color
        cloud_pd.export("tmp_corrcloud.ply")

    """' Save as relationship_*.json """
    list_relationships = list()

    relationships = gen_relationship(
        0, map_segment_pd_2_gt, instance2labelName, gt_segments_2_pd_segments
    )
    if (
        len(relationships["objects"]) != 0
        and len(relationships["relationships"]) != 0
    ):
        list_relationships.append(relationships)

    return list_relationships, segs_neighbors


def gen_relationship(
    split: int,
    map_segment_pd_2_gt: dict,
    instance2labelName: dict,
    gt_segments_2_pd_segments: dict,
    target_segments: list = None,
) -> dict:
    """' Save as relationship_*.json"""
    relationships = dict()
    relationships["scan"] = scan_id
    relationships["split"] = split

    objects = dict()
    for seg, segment_gt in map_segment_pd_2_gt.items():
        if target_segments is not None:
            if seg not in target_segments:
                continue
        name = instance2labelName[segment_gt]
        assert name != "-" and name != "none"
        objects[int(seg)] = name
    relationships["objects"] = objects

    split_relationships = list()

    """ Build "same part" relationship """
    idx_in_txt_new = 0
    for _, groups in gt_segments_2_pd_segments.items():
        if target_segments is not None:
            filtered_groups = list()
            for g in groups:
                if g in target_segments:
                    filtered_groups.append(g)
            groups = filtered_groups
        if len(groups) <= 1:
            continue

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                split_relationships.append(
                    [
                        int(groups[i]),
                        int(groups[j]),
                        idx_in_txt_new,
                        name_same_segment,
                    ]
                )
                split_relationships.append(
                    [
                        int(groups[j]),
                        int(groups[i]),
                        idx_in_txt_new,
                        name_same_segment,
                    ]
                )

    relationships["relationships"] = split_relationships
    return relationships


if __name__ == "__main__":
    args = Parser().parse_args()
    debug |= args.debug
    if debug:
        args.verbose = True
    if args.search_method == "BBOX":
        search_method = SAMPLE_METHODS.BBOX
    elif args.search_method == "KNN":
        search_method = SAMPLE_METHODS.RADIUS

    util.set_random_seed(2020)
    import json
    import os

    if args.type == "train":
        scan_ids = util.read_txt_to_list(define.SCANNET_SPLIT_TRAIN)
    elif args.type == "validation":
        scan_ids = util.read_txt_to_list(define.SCANNET_SPLIT_VAL)

    target_scan = []
    if args.target_scan != "":
        target_scan = util.read_txt_to_list(args.target_scan)

    valid_scans = list()
    relationships_new = dict()
    relationships_new["scans"] = list()
    relationships_new["neighbors"] = dict()
    counter = 0
    for scan_id in tqdm(sorted(scan_ids)):
        if len(target_scan) != 0:
            if scan_id not in target_scan:
                continue
        if debug or args.verbose:
            print(scan_id)
        relationships, segs_neighbors = process(
            define.SCANNET_DATA_PATH, scan_id, label_type=args.label_type
        )
        valid_scans.append(scan_id)
        relationships_new["scans"] += relationships
        relationships_new["neighbors"][scan_id] = segs_neighbors
        if debug:
            break

    if args.label_type == "NYU40":
        classes_json = util_label.NYU40_Label_Names
    elif args.label_type == "ScanNet20":
        classes_json = util_label.SCANNET20_Label_Names

    Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    pth_args = os.path.join(args.pth_out, "args.json")
    with open(pth_args, "w") as f:
        tmp = vars(args)
        json.dump(tmp, f, indent=2)

    pth_relationships_json = os.path.join(
        args.pth_out, "relationships_" + args.type + ".json"
    )
    with open(pth_relationships_json, "w") as f:
        json.dump(relationships_new, f)
    pth_classes = os.path.join(args.pth_out, "classes160.txt")
    with open(pth_classes, "w") as f:
        for name in classes_json:
            if name == "-":
                continue
            f.write("{}\n".format(name))
    pth_relation = os.path.join(args.pth_out, "relationships.txt")
    with open(pth_relation, "w") as f:
        f.write("{}\n".format(name_same_segment))
    pth_split = os.path.join(args.pth_out, args.type + "_scans.txt")
    with open(pth_split, "w") as f:
        for name in valid_scans:
            f.write("{}\n".format(name))
