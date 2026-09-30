"""Produce real annotated examples for all classes and detector failure cases."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .demo import load_run
from .detector import HandDetector
from .features import LABELS
from .presentation import render_overlay
from .storage import PROJECT, read_csv, write_json


def select_examples(rows):
    selected = []
    for label in LABELS:
        candidates = [row for row in rows if row["class_name"] == label and row["split"] == "test" and row["status"] == "usable"]
        if not candidates:
            raise ValueError(f"No usable test example for {label}")
        selected.append(min(candidates, key=lambda row: row["image_path"]))
    for failure in ["no_hand", "ambiguous_multiple_hands"]:
        candidates = [row for row in rows if row["split"] == "test" and row["status"] == failure]
        if candidates:
            selected.append(min(candidates, key=lambda row: row["image_path"]))
    return selected


def create_gallery(output, run=None):
    run, predictor, config = load_run(run)
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous demonstrations")
    output.mkdir(parents=True)
    selected = select_examples(read_csv(run / "detections.csv"))
    records, thumbnails = [], []
    with HandDetector(run / "hand_landmarker.task", config=config) as detector:
        for index, row in enumerate(selected):
            with Image.open(PROJECT / row["image_path"]) as source:
                rgb = np.asarray(source.convert("RGB"))
            detection = detector.detect(rgb)
            probabilities = predictor.probabilities(detection.feature)[0] if detection.feature is not None else None
            predicted = LABELS[int(probabilities.argmax())] if probabilities is not None else None
            score = float(probabilities.max()) if probabilities is not None else None
            overlay = render_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detection, probabilities, predictor.threshold,
                                     subtitle=f"Dataset example | reference: {row['class_name']}")
            filename = f"{index + 1:02d}-{row['class_name']}-{detection.status}.png"
            if not cv2.imwrite(str(output / filename), overlay):
                raise OSError("Could not save annotated example")
            # Contact sheet shows the image and landmarks, with readable captions below it.
            view = Image.fromarray(cv2.cvtColor(overlay[72:72 + rgb.shape[0], :rgb.shape[1]], cv2.COLOR_BGR2RGB))
            thumbnails.append(ImageOps.contain(view, (310, 270)))
            records.append({"reference_label": row["class_name"], "image_path": row["image_path"],
                            "original_saved_status": row["status"], "status": detection.status, "predicted_label": predicted,
                            "score": score, "accepted": bool(score is not None and score >= predictor.threshold),
                            "overlay_file": filename})
            print(f"{row['class_name']}: {predicted or detection.status}" + (f" ({score:.1%})" if score is not None else ""), flush=True)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default(size=16)
    columns, tile_w, tile_h = 4, 330, 345
    sheet = Image.new("RGB", (columns * tile_w, ((len(records) + columns - 1) // columns) * tile_h), "#161d24")
    draw = ImageDraw.Draw(sheet)
    for index, (record, thumbnail) in enumerate(zip(records, thumbnails)):
        left, top = (index % columns) * tile_w, (index // columns) * tile_h
        sheet.paste(thumbnail, (left + (tile_w - thumbnail.width) // 2, top + 8))
        draw.text((left + 10, top + 282), "Reference: " + record["reference_label"], fill="white", font=font)
        caption = record["predicted_label"] or record["status"]
        if record["score"] is not None:
            caption += f" ({record['score']:.1%})"
        draw.text((left + 10, top + 309), caption, fill="#9ceab5" if record["accepted"] else "#f8c77d", font=font)
    sheet.save(output / "contact-sheet.jpg", quality=92)
    write_json(output / "examples.json", {"run_path": str(run),
               "selection": "First usable test image per class, sorted by path; no filtering by prediction correctness or confidence. Plus first test detector failure per category.",
               "examples": records, "note": "These are dataset images, not webcam validation or aggregate evaluation."})
    lines = ["# 图片演示", "", "每类选择路径排序后的第一张可用测试图片，不按预测是否正确或分数高低挑选。另附检测失败案例。",
             "这些示例来自课程数据集，不代表摄像头实测。全测试集结果见训练运行目录 REPORT.md。", "",
             "![全部示例](contact-sheet.jpg)", "", "| 参考类别 | 实际预测/状态 | 分数 | 完整界面 |", "|---|---|---:|---|"]
    for row in records:
        score = f"{row['score']:.1%}" if row["score"] is not None else "—"
        lines.append(f"| {row['reference_label']} | {row['predicted_label'] or row['status']} | {score} | [查看]({row['overlay_file']}) |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Gallery saved: {output}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or PROJECT / "reports" / datetime.now().strftime("gallery-%Y%m%d-%H%M%S")
    create_gallery(output.resolve(), args.run_dir)


if __name__ == "__main__":
    main()
