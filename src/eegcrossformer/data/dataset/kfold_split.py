import numpy as np
import torch


def kfold_split(
  dataset: torch.utils.data.Dataset,
  fold: int,
  kfold: int,
  random_seed: int,
  val_ratio: float = 0.2,
):
  fold_size = len(dataset) // kfold
  indices = torch.randperm(
    len(dataset), generator=torch.Generator().manual_seed(random_seed)
  ).tolist()
  test_indices = indices[(fold - 1) * fold_size : fold * fold_size]
  train_indices = [i for i in indices if i not in test_indices]
  split_point = int(len(train_indices) * (1 - val_ratio))
  val_indices = train_indices[split_point:]
  train_indices = train_indices[:split_point]

  train_set = torch.utils.data.Subset(dataset, train_indices)
  validation_set = torch.utils.data.Subset(dataset, val_indices)
  test_set = torch.utils.data.Subset(dataset, test_indices)

  return train_set, validation_set, test_set


def _get_labels(dataset):
  """Extract labels from a dataset or ConcatDataset."""
  if hasattr(dataset, "labels"):
    return np.asarray(dataset.labels)
  if isinstance(dataset, torch.utils.data.ConcatDataset):
    parts = [_get_labels(d) for d in dataset.datasets]
    return np.concatenate(parts)
  raise ValueError(f"Cannot extract labels from {type(dataset)}")


def stratified_kfold_split(
  dataset: torch.utils.data.Dataset,
  fold: int,
  kfold: int,
  random_seed: int,
  val_ratio: float = 0.2,
):
  """K-fold split that ensures each fold has balanced class distribution."""
  labels = _get_labels(dataset)
  n = len(dataset)
  rng = np.random.RandomState(random_seed)

  # Group indices by class, shuffle within each class
  unique_classes = np.unique(labels)
  class_indices = {}
  for c in unique_classes:
    idx = np.where(labels == c)[0]
    rng.shuffle(idx)
    class_indices[c] = idx

  # Assign each class's indices into kfold folds (round-robin)
  folds = [[] for _ in range(kfold)]
  for c in unique_classes:
    for i, idx in enumerate(class_indices[c]):
      folds[i % kfold].append(idx)

  # Shuffle within each fold
  for f in folds:
    rng.shuffle(f)

  test_indices = folds[fold - 1]
  train_val_indices = []
  for i in range(kfold):
    if i != fold - 1:
      train_val_indices.extend(folds[i])
  rng.shuffle(train_val_indices)

  split_point = int(len(train_val_indices) * (1 - val_ratio))
  train_indices = train_val_indices[:split_point]
  val_indices = train_val_indices[split_point:]

  train_set = torch.utils.data.Subset(dataset, train_indices)
  validation_set = torch.utils.data.Subset(dataset, val_indices)
  test_set = torch.utils.data.Subset(dataset, test_indices)

  return train_set, validation_set, test_set