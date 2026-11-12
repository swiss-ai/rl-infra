"""External NeMo-RL dataset and processor for parallel translation data."""

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path

from .prompts import format_example


class TranslationDataset(RawDataset):
    def __init__(self, data_path, split="train", split_validation_size=0,
                 seed=42, **kwargs):
        self.task_name = "mt"
        raw = load_dataset_from_path(data_path, data_split=split)
        self.dataset = raw.map(
            lambda row, index: {**format_example(row, index, seed), "task_name": self.task_name},
            with_indices=True, remove_columns=raw.column_names)
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

