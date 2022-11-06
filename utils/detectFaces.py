#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# A simple implementation of face tracking leveraging a face detector. Using other advanced face tracker of your choice can potentially lead to better separation results.

import argparse
import os

import crop_mouth_from_video
import cv2
import face_alignment
import mmcv
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

import utils


def face2head(box, scale=1.5):
    if box is None:
        return box
    width = box[2] - box[0]
    height = box[3] - box[1]
    width_center = (box[2] + box[0]) / 2
    height_center = (box[3] + box[1]) / 2
    square_width = int(max(width, height) * scale)
    return [
        width_center - square_width / 2,
        height_center - square_width / 2,
        width_center + square_width / 2,
        height_center + square_width / 2,
    ]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--detect_every_N_frame", type=int, default=8)
    parser.add_argument("--scalar_face_detection", type=float, default=1.5)
    parser.add_argument("--number_of_speakers", type=int, default=2)
    parser.add_argument("--vertical_split", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))
    utils.mkdirs(os.path.join(args.output_path, "faces"))

    landmarks_dic = {}
    faces_dic = {}
    boxes_dic = {}
    for i in range(args.number_of_speakers):
        landmarks_dic[i] = []
        faces_dic[i] = []
        boxes_dic[i] = []

    mtcnn = MTCNN(keep_all=True, device=device, select_largest=True)

    video = mmcv.VideoReader(args.video_input_path)
    width = video.width
    height = video.height
    print("Video statistics: ", width, height, video.fps)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device=device.type
    )

    for iframe, frame in enumerate(video):
        print("\rTracking frame: {}".format(iframe + 1), end="")
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect faces
        if iframe % args.detect_every_N_frame == 0:
            if args.number_of_speakers == 2 and args.vertical_split > 0:
                # ensure faces are on either side of the split-screen border
                split = width * args.vertical_split
                boxes = mtcnn.detect(frame)[0]
                if boxes is None:
                    boxes = []
                left = None
                right = None
                # take the largest face on either side
                for box in boxes:
                    position = (box[0] + box[2]) / 2
                    if position < split and left is None:
                        left = box
                    elif position >= split and right is None:
                        right = box
                if left is None and not boxes_dic[0]:
                    left = [split / 3, height / 3, split / 3 * 2, height / 3 * 2]
                if right is None and not boxes_dic[1]:
                    right = [
                        split + split / 3,
                        height / 3,
                        split + split / 3 * 2,
                        height / 3 * 2,
                    ]
                boxes = [left, right]
            else:
                boxes = mtcnn.detect(frame)[0]
            boxes = boxes[: args.number_of_speakers]
            boxes = [face2head(box, args.scalar_face_detection) for box in boxes]
        else:
            boxes = [boxes_dic[j][-1] for j in range(args.number_of_speakers)]

        # Crop faces and save landmarks for each speaker
        if len(boxes) != args.number_of_speakers:
            boxes = [None] * args.number_of_speakers

        for j, box in enumerate(boxes):
            if box is None:
                box = boxes_dic[j][-1]

            face = frame.crop((box[0], box[1], box[2], box[3])).resize((224, 224))
            preds = fa.get_landmarks(np.array(face))
            if iframe == 0:
                faces_dic[j].append(face)
                landmarks_dic[j].append(preds)
                boxes_dic[j].append(box)
            else:
                iou_scores = []
                for b_index in range(args.number_of_speakers):
                    last_box = boxes_dic[b_index][-1]
                    iou_score = bb_intersection_over_union(box, last_box)
                    iou_scores.append(iou_score)
                box_index = iou_scores.index(max(iou_scores))

                other = 1 - box_index
                if args.number_of_speakers == 2 and len(landmarks_dic[box_index]) > len(
                    landmarks_dic[other]
                ):
                    # do not assign two boxes to the same speaker
                    box_index = other

                faces_dic[box_index].append(face)
                landmarks_dic[box_index].append(preds)
                boxes_dic[box_index].append(box)

    # Save landmarks
    for i in range(args.number_of_speakers):
        landmarks_dic[i] = [
            [x]
            for x in crop_mouth_from_video.landmarks_interpolate(
                [xs if xs is None else xs[0] for xs in landmarks_dic[i]]
            )
        ]
        utils.save2npz(
            os.path.join(args.output_path, "landmark", "speaker" + str(i + 1) + ".npz"),
            data=landmarks_dic[i],
        )
        dim = face.size
        fourcc = cv2.VideoWriter_fourcc(*"FMP4")
        speaker_video = cv2.VideoWriter(
            os.path.join(args.output_path, "faces", "speaker" + str(i + 1) + ".mp4"),
            fourcc,
            25.0,
            dim,
        )
        for frame in faces_dic[i]:
            speaker_video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        speaker_video.release()

    # Output video path
    parts = args.video_input_path.split("/")
    video_name = parts[-1][:-4]
    if not os.path.exists(os.path.join(args.output_path, "filename_input")):
        os.mkdir(os.path.join(args.output_path, "filename_input"))
    csvfile = open(
        os.path.join(args.output_path, "filename_input", str(video_name) + ".csv"), "w"
    )
    for i in range(args.number_of_speakers):
        csvfile.write("speaker" + str(i + 1) + ",0\n")
    csvfile.close()


if __name__ == "__main__":
    main()
