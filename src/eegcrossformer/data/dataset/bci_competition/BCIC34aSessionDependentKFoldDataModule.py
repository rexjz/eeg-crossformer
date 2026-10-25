from typing import Literal, Optional
from eegcrossformer.data.dataset.bci_competition.cache_utils import (
  BCIC34aSubjectCachedDataset,
  load_or_build_bcic34a_subject_cache,
)
from eegcrossformer.data.dataset.bci_competition.BCIC3Dataset4a import (
  BcicompetitionIIIDatasetIVaRawTrailDataset,
)
from eegcrossformer.project_config import bcic3_dataset4a_path
from torch.utils.data import random_split
import lightning as L
import random
import torch
import torcheeg.transforms as transforms


class BCIC34aSessionDependentKFoldDataModule(L.LightningDataModule):
  def __init__(
    self,
    data_dir: str = bcic3_dataset4a_path,
    subject_code="aa",
    batch_size=64,
    random_seed=42,
    kfold=5,
    fold=1,
    num_workers=4,
    transform: Optional[transforms.EEGTransform] = None,
    get_item_style: Literal["dict", "tuple"] = "dict",
    small_training_mode: bool = False,
    use_cache: bool = True,
    force_refresh_cache: bool = False,
  ):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.random_seed = random_seed
    self.subject_code = subject_code
    self.num_workers = num_workers
    self.kfold = kfold
    self.fold = fold
    self.transform = transform
    self.get_item_style = get_item_style
    self.small_training_mode = small_training_mode
    self.use_cache = use_cache
    self.force_refresh_cache = force_refresh_cache
    self.cache_entry = None

  def setup(self, stage: str):
    if self.use_cache:
      self.cache_entry = load_or_build_bcic34a_subject_cache(
        subject_code=self.subject_code,
        transform=self.transform,
        root_path=self.data_dir,
        force_refresh=self.force_refresh_cache,
      )
      dataset = BCIC34aSubjectCachedDataset(
        self.cache_entry.data,
        self.cache_entry.labels,
        get_item_style=self.get_item_style,
      )
    else:
      # 100Hz 350 in length
      dataset = BcicompetitionIIIDatasetIVaRawTrailDataset(
        root_path=self.data_dir,
        subject_code=self.subject_code,
        transform=self.transform,
        get_item_style=self.get_item_style,
      )
    all_ids = [x for x in range(0, len(dataset))]
    local_rng = random.Random(self.random_seed)
    local_rng.shuffle(all_ids)
    division_point = int(len(all_ids) / self.kfold)

    if self.small_training_mode:
      # 20% train / 80% test (small-training switch)
      testing_ids = all_ids[0 : division_point * (self.fold - 1)]
      testing_ids.extend(all_ids[division_point * self.fold :])
      training_ids = all_ids[
        division_point * (self.fold - 1) : division_point * self.fold
      ]
    else:
      # 80% train / 20% test (default)
      training_ids = all_ids[0 : division_point * (self.fold - 1)]
      training_ids.extend(all_ids[division_point * self.fold :])
      testing_ids = all_ids[division_point * (self.fold - 1) : division_point * self.fold]

    training_dataset = torch.utils.data.Subset(dataset=dataset, indices=training_ids)
    testing_set = torch.utils.data.Subset(dataset=dataset, indices=testing_ids)

    if self.small_training_mode:
      # Too few training samples to split off a validation set
      training_set = training_dataset
      validation_set = testing_set  # placeholder; trainer will skip val via limit_val_batches=0
    else:
      training_set, validation_set = random_split(
        training_dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(self.random_seed),
      )

    if stage == "fit":
      self.train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
      self.validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=self.batch_size, num_workers=self.num_workers
      )
    if stage == "test":
      self.test_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=self.batch_size, num_workers=self.num_workers
      )

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.validation_loader

  def test_dataloader(self):
    return self.test_loader
