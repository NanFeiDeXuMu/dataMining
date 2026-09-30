"""Train with validation-only early stopping; evaluate frozen models on test once."""

import argparse
import copy
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "matplotlib-cache"))

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

from .audit import sha256
from .config import TrainingConfig
from .features import FEATURE_VERSION, LABELS
from .metrics import evaluate_predictions
from .network import make_classifier
from .predictor import Predictor, fit_standardizer
from .storage import PROJECT, contained_run, read_csv, read_json, status, write_csv, write_json


def fit_mlp(train_x, train_y, validation_x, validation_y, config):
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    model = make_classifier().cpu()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    val_x, val_y = torch.from_numpy(validation_x), torch.from_numpy(validation_y)
    history, best_state, best_loss, best_epoch, stale = [], None, float("inf"), 0, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch in torch.randperm(len(train_x)).split(config.batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_logits, val_logits = model(train_x), model(val_x)
            train_loss, val_loss = float(loss_fn(train_logits, train_y)), float(loss_fn(val_logits, val_y))
            train_accuracy = float((train_logits.argmax(1) == train_y).float().mean())
            val_accuracy = float((val_logits.argmax(1) == val_y).float().mean())
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("Non-finite epoch loss")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "train_accuracy": train_accuracy, "validation_accuracy": val_accuracy})
        if val_loss < best_loss:
            improvement = best_loss - val_loss
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
            stale = 0 if improvement >= config.min_delta else stale + 1
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.2%}", flush=True)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"selected_epoch": best_epoch, "validation_loss": best_loss,
                            "selection_rule": "minimum validation cross-entropy; test not used",
                            "epochs_ran": len(history)}


def save_plots(run, history, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = [row["epoch"] for row in history]
    for key, label in [("train", "Training"), ("validation", "Validation")]:
        axes[0].plot(epochs, [row[f"{key}_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row[f"{key}_accuracy"] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy on usable images", title="Training and validation accuracy", ylim=(0, 1.02))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "training_curves.png", dpi=160)
    plt.close(fig)
    cm = np.asarray(metrics["confusion_matrix_with_detection_failure"])
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(19), xticklabels=[*LABELS, "Detection failure"], yticks=range(18), yticklabels=LABELS,
           xlabel="Predicted label", ylabel="True label", title=f"Test confusion matrix (all {metrics['original_images']:,} original test images)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Images")
    fig.tight_layout()
    fig.savefig(run / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def write_report(run, summary, selection):
    mlp, baseline = summary["test"]["mlp"], summary["test"]["logistic_regression"]
    extraction = read_json(run / "extraction.json")
    lines = ["# 手势识别实验报告", "", f"运行：`{run.name}`。全部指标来自该目录保存的产物。", "",
             "## 数据与方法", "",
             f"原始图片 {extraction['original_images']:,} 张，18 类；可用关键点特征 {extraction['usable_images']:,} 张。",
             "原始图片按类分层随机划分 70% / 15% / 15%，种子 42；检测失败仍计入原始评价分母。",
             "清单缺少人员 ID，本实验不能证明跨人员独立泛化。未做近重复图像识别。",
             "MediaPipe 预训练模型不参与训练。MLP 使用 63 维方向保留特征、训练集标准化及 128/64 隐藏层。",
             f"验证 loss 最小的第 {selection['selected_epoch']} 轮被选中；共运行 {selection['epochs_ran']} 轮。",
             "测试集只在模型选择冻结后评价。Logistic Regression 为固定参数比较基线。", "",
             "## 测试结果", "", "| 指标 | MLP | Logistic Regression |", "|---|---:|---:|"]
    for label, key in [("可用样本分类准确率", "accuracy_on_usable"), ("全部原始图片端到端正确率", "end_to_end_accuracy"),
                       ("可用样本 Macro-F1", "macro_f1_on_usable")]:
        lines.append(f"| {label} | {mlp[key]:.2%} | {baseline[key]:.2%} |")
    lines += ["", f"测试分母：原始 {mlp['original_images']} 张，可用 {mlp['usable_images']} 张；MLP 正确 {mlp['correct_predictions']} 张。",
              f"检测/特征覆盖率：{mlp['feature_coverage']:.2%}。", "",
              "无手、多手歧义和无效几何均计为端到端失败，不被静默移除。",
              f"演示拒识阈值固定为 {mlp['confidence_threshold']:.2f}；softmax 分数未校准，低于阈值显示 Uncertain。",
              "上面的总体分类指标采用最高分标签，不应用演示阈值或时间平滑；拒识后的覆盖率与准确率另存 evaluation.json。", "",
              "## 证据", "", "![训练曲线](training_curves.png)", "", "![测试混淆矩阵](confusion_matrix.png)", "",
              "逐图预测见 predictions.csv；逐类指标见 per_class_test.csv；失败状态见 detections.csv。",
              "source/ 保存本次源码与锁定依赖，config.json 保存算法参数，artifacts.json 记录核心产物哈希。", "",
              "## 边界", "", "静态图片评价不等于摄像头实测。mute 的嘴唇上下文不在手部关键点中。",
              "掌心/手背、左右手、遮挡和光照变化需在现场分别验证；本报告不把关联或置信度解释为可靠概率。"]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_run(run, project=PROJECT):
    run = contained_run(run, project)
    if (run / "training_started.json").exists():
        raise ValueError("This run has already started training; create a new run instead of reusing its test set")
    config_dict = read_json(run / "config.json")
    config = TrainingConfig(**config_dict["training"])
    with np.load(run / "features.npz", allow_pickle=False) as stored:
        x, y, usable, splits = (stored[key].copy() for key in ["features", "labels", "usable", "splits"])
        paths = stored["image_paths"].copy()
        if str(stored["feature_version"]) != FEATURE_VERSION:
            raise ValueError("Feature version mismatch")
    train_mask, val_mask = usable & (splits == "train"), usable & (splits == "validation")
    mean, scale = fit_standardizer(x[train_mask])
    write_json(run / "training_started.json", {"config": asdict(config), "test_accessed": False})
    status(run, "training")
    started = time.perf_counter()
    try:
        source = run / "source" / "gesture_project"
        source.mkdir(parents=True)
        for path in (project / "gesture_project").glob("*.py"):
            shutil.copyfile(path, source / path.name)
        shutil.copyfile(project / "requirements.lock.txt", run / "source" / "requirements.lock.txt")
        model, history, selection = fit_mlp((x[train_mask] - mean) / scale, y[train_mask],
                                            (x[val_mask] - mean) / scale, y[val_mask], config)
        write_csv(run / "training_history.csv", history)
        write_json(run / "selection.json", selection)
        checkpoint = {"purpose": "trained_gesture_classifier", "state_dict": model.state_dict(),
                      "labels": list(LABELS), "feature_version": FEATURE_VERSION,
                      "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                      "confidence_threshold": config.confidence_threshold, "selected_epoch": selection["selected_epoch"]}
        torch.save(checkpoint, run / "classifier.pt")
        predictor = Predictor(run / "classifier.pt")
        with torch.inference_mode():
            expected = model(torch.from_numpy((x[val_mask] - mean) / scale)).softmax(1).numpy()
        np.testing.assert_allclose(predictor.probabilities(x[val_mask]), expected, atol=1e-6, rtol=1e-6)

        baseline = LogisticRegression(max_iter=2000, C=1.0, random_state=config.seed)
        baseline.fit((x[train_mask] - mean) / scale, y[train_mask])
        joblib.dump({"model": baseline, "mean": mean, "scale": scale}, run / "baseline.joblib")
        # No test features have been predicted or used for model selection above this line.
        summary, prediction_rows = {}, []
        detection_rows = read_csv(run / "detections.csv")
        for split in ["validation", "test"]:
            mask = splits == split
            selected = np.flatnonzero(mask)
            good_indices = np.flatnonzero(mask & usable)
            mlp_probabilities = predictor.probabilities(x[good_indices])
            baseline_probabilities = baseline.predict_proba((x[good_indices] - mean) / scale)
            positions = {int(index): position for position, index in enumerate(good_indices)}
            predicted = {name: np.full(len(selected), -1, dtype=int) for name in ["mlp", "logistic_regression"]}
            confidence = {name: np.zeros(len(selected)) for name in predicted}
            for position, index in enumerate(selected):
                row = {"image_path": str(paths[index]), "split": split, "true_class_id": int(y[index]),
                       "true_label": LABELS[y[index]], "status": detection_rows[index]["status"]}
                for name, probabilities in [("mlp", mlp_probabilities), ("logistic_regression", baseline_probabilities)]:
                    if usable[index]:
                        values = probabilities[positions[int(index)]]
                        predicted[name][position], confidence[name][position] = int(values.argmax()), float(values.max())
                    label_id = int(predicted[name][position])
                    row[f"{name}_class_id"] = label_id
                    row[f"{name}_label"] = LABELS[label_id] if label_id >= 0 else ""
                    row[f"{name}_confidence"] = float(confidence[name][position])
                    row[f"{name}_correct"] = bool(label_id == y[index])
                prediction_rows.append(row)
            summary[split] = {name: evaluate_predictions(y[mask], predicted[name], confidence[name], config.confidence_threshold)
                              for name in predicted}
        summary["protocol"] = {"test_used_for_selection": False, "split_unit": "image, stratified by class",
                               "person_independence_verified": False, "checkpoint_roundtrip_passed": True,
                               "train_samples": int(train_mask.sum()), "validation_samples": int(val_mask.sum()),
                               "elapsed_seconds": time.perf_counter() - started,
                               "versions": {name: importlib.metadata.version(name) for name in
                                            ["torch", "mediapipe", "numpy", "scikit-learn", "opencv-contrib-python"]}}
        write_json(run / "evaluation.json", summary)
        write_csv(run / "predictions.csv", prediction_rows)
        write_csv(run / "per_class_test.csv", summary["test"]["mlp"]["per_class"])
        save_plots(run, history, summary["test"]["mlp"])
        write_report(run, summary, selection)
        core = ["classifier.pt", "baseline.joblib", "hand_landmarker.task", "config.json", "features.npz", "data_audit.json", "per_class_test.csv",
                "split_manifest.csv", "detections.csv", "predictions.csv", "evaluation.json", "extraction.json",
                "training_history.csv", "selection.json", "training_curves.png", "confusion_matrix.png", "REPORT.md"]
        core.extend(path.relative_to(run).as_posix() for path in (run / "source").rglob("*") if path.is_file())
        write_json(run / "artifacts.json", {"sha256": {name: sha256(run / name) for name in core}})
        status(run, "trained")
        from .validate import publish_run
        publish_run(run, project)
        print(f"Training complete and activated: {run}", flush=True)
        print(f"Test usable accuracy={summary['test']['mlp']['accuracy_on_usable']:.2%}; "
              f"end-to-end={summary['test']['mlp']['end_to_end_accuracy']:.2%}", flush=True)
        return summary
    except Exception as exc:
        status(run, "failed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    train_run(args.run_dir)


if __name__ == "__main__":
    main()
