#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: str) -> None:
    print("\n", cmd, "\n")
    with subprocess.Popen(shlex.split(cmd)) as process:
        pass
    assert process.returncode == 0, cmd


def preprocess_video(
    video: Path,
    tmp: Path,
    start_height: float,
    start_width: float,
    stop_height: float,
    stop_width: float,
) -> tuple[Path, Path]:
    # video at 25fps
    video_out = tmp / "output.mp4"
    filter_cmd = "fps=fps=25"
    # option to crop
    width = stop_width - start_width
    height = stop_height - start_height
    if width < 1 or height < 1:
        filter_cmd += f",crop=w=in_w*{width}:h=in_h*{height}:x=in_w*{start_width}:y=in_h*{start_height}"
    cmd = f"ffmpeg -i {video} -filter:v {filter_cmd} {video_out}"
    if not video_out.is_file():
        run(cmd)

    # process audio
    audio_out = video_out.with_suffix(".wav")
    cmd = f"ffmpeg -i {video_out} -vn -ar 16000 -ac 1 -ab 192k -f wav {audio_out}"
    if not audio_out.is_file():
        run(cmd)
    return video_out, audio_out


def detect_faces(video: Path, tmp: Path, vertical_split: float) -> None:
    cmd = f"{sys.executable} ./utils/detectFaces.py --video_input_path {video} --output_path {tmp} --number_of_speakers 2 --scalar_face_detection 1.5 --detect_every_N_frame 8 --vertical_split {vertical_split}"
    run(cmd)


def crop_mouth(video: Path, tmp: Path) -> None:
    cmd = f"{sys.executable} ./utils/crop_mouth_from_video.py --video-direc {tmp/'faces'}/ --landmark-direc {tmp/'landmark'}/ --save-direc {tmp/'mouthroi'}/ --convert-gray --filename-path {tmp/'filename_input'/video.with_suffix('.csv').name}"
    run(cmd)


def separate(audio: Path, tmp: Path) -> None:
    cmd = f"{sys.executable} testRealVideo.py --mouthroi_root {tmp/'mouthroi'}/ --facetrack_root {tmp/'faces'}/ --audio_path {audio} --weights_lipreadingnet lipreading_best.pth --weights_facial facial_best.pth --weights_unet unet_best.pth --weights_vocal vocal_best.pth --lipreading_config_path configs/lrw_snv1x_tcn2x.json --num_frames 64 --audio_length 2.55 --hop_size 160 --window_size 400 --n_fft 512 --unet_output_nc 2 --normalization --visual_feature_type both --identity_feature_dim 128 --audioVisual_feature_dim 1152 --visual_pool maxpool --audio_pool maxpool --compression_type none --reliable_face --number_of_speakers 2 --mask_clip_threshold 5 --hop_length 2.55 --lipreading_extract_feature --number_of_identity_frames 1 --output_dir_root {tmp}/ --checkpoints_dir {tmp/'checkpoints'}/"
    run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to the video")
    parser.add_argument(
        "--output",
        type=str,
        help="Folder where to write the output (and temporary data)",
    )
    parser.add_argument(
        "--vertical_split",
        type=float,
        default=0.5,
        help="Percentage of the frame width to indicate the side-by-side split. "
        "A negative value assumes no side-by-side split. This option is applied "
        "AFTER --crop*",
    )
    parser.add_argument(
        "--crop_start_height",
        type=float,
        default=0.0,
        help="Percentage of the frame height where the cropping starts.",
    )
    parser.add_argument(
        "--crop_stop_height",
        type=float,
        default=1.0,
        help="Percentage of the frame height where the cropping stops.",
    )
    parser.add_argument(
        "--crop_start_width",
        type=float,
        default=0.0,
        help="Percentage of the frame width where the cropping starts.",
    )
    parser.add_argument(
        "--crop_stop_width",
        type=float,
        default=1.0,
        help="Percentage of the frame width where the cropping stops.",
    )
    args = parser.parse_args()

    video = Path(args.video).resolve()
    assert video.is_file()

    out = Path(args.output).resolve()
    assert out.is_dir()

    # does the result already exist?
    result = out / video.with_suffix(".wav").name.replace(".wav", "_0.wav")
    if result.is_file():
        sys.exit(0)

    # create temporary folder
    out = out / video.with_suffix("").name
    out.mkdir(exist_ok=True)

    # change folder
    os.chdir(Path(__file__).parent.resolve())

    video, audio = preprocess_video(
        video,
        out,
        args.crop_start_height,
        args.crop_start_width,
        args.crop_stop_height,
        args.crop_stop_width,
    )
    detect_faces(video, out, args.vertical_split)
    crop_mouth(video, out)
    separate(audio, out)

    for ispeaker, speaker in enumerate(sorted(out.glob("speaker*.wav"))):
        speaker.rename(
            result.parent / result.name.replace("_0.wav", f"_{ispeaker}.wav")
        )
    shutil.rmtree(out)
