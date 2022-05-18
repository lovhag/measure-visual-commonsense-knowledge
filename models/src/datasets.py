import numpy as np
import json
import h5py
from torch.utils.data import Dataset

def get_text_image_pretraining_dataset(train_path: str, val_path: str, tokenizer, image_features_path: str):
    train_ds = TextImageDataset(train_path, tokenizer, image_features_path)
    val_ds = TextImageDataset(val_path, tokenizer, image_features_path)
    return train_ds, val_ds


class TextImageDataset(Dataset):

    def __init__(self, examples_path: str, tokenizer, image_features_path: str=None):
        """A dataset for pretraining using text and optionally precomputed image features

        Args:
            examples_path (str): Path to jsonl file with fields "text", "answer" and optionally "image_id"
            tokenizer (BertTokenizer): Tokenizer to tokenize text
            image_features_path (str, optional): Path to hdf5 file with "features" and "ids" datasets. Defaults to None.
        """
        super().__init__()

        self.tokenizer = tokenizer

        # Load examples set
        self.examples = [json.loads(line) for line in open(examples_path)]

        # Load image features
        if image_features_path is not None:
            buffer = h5py.File(image_features_path, mode="r")
            self.image_features = buffer["features"]
            self.image_id2idx = {id: idx for idx, id in enumerate(buffer["ids"].asstr())}
        else:
            self.image_features = self.image_id2idx = None
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        preprocessed_example = self.tokenizer(example["text"], truncation=True)
        if "image_id" in example and self.image_features is not None:
            preprocessed_example["img_feats"] = np.array(self.image_features[self.image_id2idx[example["image_id"]]], dtype=np.float16)[np.newaxis, ...]
        return preprocessed_example
