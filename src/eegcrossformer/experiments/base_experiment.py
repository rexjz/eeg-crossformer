from abc import ABC, abstractmethod
from datetime import datetime
import os
import lightning as L
import torch


class Experiment(ABC):
  def __init__(self, batch_size=256):
    torch.set_float32_matmul_precision("high")
    self.model_name = self.get_model_name()
    self.now = datetime.now()
    self.date_time = self.now.strftime("%Y.%m.%d-%H.%M.%S")
    # The process id keeps concurrent runs started in the same second apart.
    self.experiment_tag = (
      f"{self.model_name}_{self.__class__.__name__}_{self.date_time}_pid{os.getpid()}"
    )
    self.batch_size = batch_size
    self.checkpoint_name = f"{self.model_name}-{self.__class__.__name__}-{{epoch:02d}}-{{val_accuracy:.4f}}-{{val_loss:.4f}}"

  @abstractmethod
  def get_model_name(self) -> str:
    """Return the name of the model being used"""
    pass

  @abstractmethod
  def create_model(self, model_kwargs: dict) -> L.LightningModule:
    """create an new model instance"""
    pass

  @abstractmethod
  def run(self):
    """Run the experiment"""
    pass
