#!/usr/bin/env python3
"""Train a binary brain tumor classifier on MRI images."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {"train": train_tfms, "val": val_tfms}


def build_loaders(
    data_dir: Path,
    batch_size: int,
    image_size: int,
    val_split: float,
    num_workers: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    transforms_map = build_transforms(image_size)
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transforms_map["train"])
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset = copy.copy(val_ds.dataset)
    val_ds.dataset.transform = transforms_map["val"]

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, dataset.classes


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print(f"{phase} epoch {epoch}/{epochs} loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain tumor MRI binary classifier training pipeline.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to MRI images (ImageFolder layout).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"), help="Output model path.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = build_loaders(
        args.data_dir, args.batch_size, args.image_size, args.val_split, args.num_workers
    )
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes for binary classification, found {len(classes)}: {classes}")

    model = build_model(num_classes=len(classes))
    model = model.to(device)
    model = train(model, loaders, device, args.epochs, args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": classes, "image_size": args.image_size},
        args.output,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
