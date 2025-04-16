"""Evaluate on standard classification webdatasets."""

import os

import open_clip
import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc
from sklearn.metrics import balanced_accuracy_score

def get_model_transforms(model_arch, model, model_path):
    assert model is not None or model_path is not None

    if model is None:
        model, transform, device = create_model(model_arch, model_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pp_cfg = open_clip.transform.PreprocessCfg(**{'size': (224, 224), 'mode': 'RGB', 'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711), 'interpolation': 'bicubic', 'resize_mode': 'shortest', 'fill_color': 0})
        transform = open_clip.transform.image_transform_v2(
            pp_cfg,
            is_train=False,
        )

    return model, transform, device

def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = open_clip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    # model.half()
    model = model.to(device)

    return model, transform, device


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path=None,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    return_preds=False,
    return_topk=False,
    model=None
):
    """Evaluate CLIP model on classification task."""

    # Create model
    model, transform, device = get_model_transforms(model_arch=model_arch, model=model, model_path=model_path)

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    logits, target = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )
    with torch.no_grad():
        pred = logits.argmax(axis=1).cpu()
        target = target.cpu()

    # Compute metrics
    if len(dataset.classes) >= 5:
        acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    else:
        (acc1,) = zsc.accuracy(logits, target, topk=(1,))
        acc5 = None
    mean_per_class_recall = balanced_accuracy_score(target, pred)
    metrics = {
        "acc1": acc1,
        "acc5": acc5,
        "mean_per_class_recall": mean_per_class_recall,
    }

    if return_preds:
        if return_topk:
            with torch.no_grad():
                _, topk_pred = torch.topk(logits, int(return_topk), dim=1)
                topk_pred = topk_pred.cpu()
            return metrics, topk_pred, target
        return metrics, pred, target
    return metrics
