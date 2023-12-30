import os

import evaluate
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import Subset
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
from transformers import Trainer
from transformers import TrainingArguments


class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, id2color, train=True, train_val_split=0.8):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.id2color = id2color

        self.img_dir = os.path.join(self.root_dir, "rgb")
        self.ann_dir = os.path.join(self.root_dir, "GT_color")

        self.images, self.annotations = self._load_data()
        self.train_indices, self.val_indices = train_test_split(range(len(self.images)), train_size=train_val_split,
                                                                random_state=42) if train else ([], [])

    def _load_data(self):
        image_file_names = sorted(os.listdir(self.img_dir))
        annotation_file_names = sorted(os.listdir(self.ann_dir))
        assert len(image_file_names) == len(annotation_file_names), "Number of images must match number of annotations"
        return image_file_names, annotation_file_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        annotation = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        # make 2D segmentation map (based on 3D one) thanks a lot, Stackoverflow:
        # https://stackoverflow.com/questions/61897492/finding-the-number-of-pixels-in-a-numpy-array-equal-to-a-given
        # -color
        annotation = np.array(annotation)
        annotation_2d = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.uint8)  # height, width

        for id, color in self.id2color.items():
            annotation_2d[(annotation == color).all(axis=-1)] = id

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = self.feature_extractor(image, Image.fromarray(annotation_2d), return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs

    def _create_dataset(self, images, annotations):
        # Common data configuration
        data = {"root_dir": self.root_dir,
                "feature_extractor": self.feature_extractor,
                "id2color": self.id2color,
                "train_val_test_split": (0.6, 0.2, 0.2)  # Split ratios for train, validation, and test
                }

        # Creating dataset for each split
        return DatasetDict({
            "train": SemanticSegmentationDataset(**data, split="train"),
            "validation": SemanticSegmentationDataset(**data, split="validation"),
            "test": SemanticSegmentationDataset(**data, split="test")
        })


class Segmentation:
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, number_of_labels, batch_size, epoch_number, lr=0.0001, ):
        self.id2label = None
        self.number_of_labels = number_of_labels
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "rgb")
        self.ann_dir = os.path.join(self.root_dir, "GT_color")
        self.pretrained_model = None
        self.optimizer = None
        self.batch_size = batch_size
        self.epoch_number = epoch_number
        self.lr = lr
        self.trainer = None
        self.train_arg = None
        self.metric = evaluate.load('mean_iou')
        #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.color_map = None
        self.color_map_path = os.path.join(self.root_dir, "colormap.txt")
        self.images = None
        self.annotations = None
        self.feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
        self.read_rgb_files()
        self.read_annotation_files()
        self.read_color_map()
        self.test_ds = None
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def set_training_arguments(self):
        self.train_arg = TrainingArguments("model_outdir/", learning_rate=self.lr, num_train_epochs=self.epoch_number,
                                           per_device_train_batch_size=self.batch_size, per_device_eval_batch_size=8,
                                           save_total_limit=3, do_eval=True, do_train=True, evaluation_strategy="epoch",
                                           save_strategy="epoch", save_steps=1, eval_steps=1,
                                           load_best_model_at_end=True, )

    def set_trainer(self, train_ds,val_ds):
        self.trainer = Trainer(model=self.pretrained_model, args=self.train_arg, train_dataset=train_ds,
                               eval_dataset=val_ds, compute_metrics=self.compute_metrics, )

    def __len__(self):
        return len(self.images)

    def read_rgb_files(self):
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)

        self.images = sorted(image_file_names)

    def read_annotation_files(self):
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

    def read_color_map(self):
        self.color_map = pd.read_csv(self.color_map_path, sep="\t", header=None)

    def arrange_label_id(self):

        self.color_map.columns = ["Class", "R", "G", "B", "ID"]
        label2id = {label: id for id, label in enumerate(self.color_map.Class)}
        id2label = {id: label for id, label in enumerate(self.color_map.Class)}
        id2color = {id: [R, G, B] for id, (R, G, B) in
                    enumerate(zip(self.color_map.R, self.color_map.G, self.color_map.B))}
        print(id2label)
        print(label2id)
        print(id2color)
        return label2id, id2label, id2color

    def preprocessing(self):
        label2id, self.id2label, id2color = self.arrange_label_id()
        self.download_pre_train_model(self.id2label, label2id)
        self.optimizer = torch.optim.AdamW(self.pretrained_model.parameters(), lr=self.lr)

        dataset = SemanticSegmentationDataset(root_dir=self.root_dir, feature_extractor=self.feature_extractor,
                                              id2color=id2color)

        # Splitting the dataset
        indices = list(range(len(dataset)))
        train_size = 0.6
        test_val_size = 0.4
        val_size = 0.5  # Half of test_val_size, which makes it 20% of total

        # Primary split between train and test_val
        train_indices, test_val_indices = train_test_split(indices, test_size=test_val_size, random_state=42)

        # Further split test_val into validation and test
        val_indices, test_indices = train_test_split(test_val_indices, test_size=val_size, random_state=42)

        # Creating subsets for train, validation, and test
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        self.test_ds = Subset(dataset, test_indices)

        self.set_training_arguments()
        self.set_trainer(train_ds=train_ds, val_ds=val_ds)

    def download_pre_train_model(self, id2label, label2id):
        self.pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b4-finetuned-ade-512-512", num_labels=self.number_of_labels, id2label=id2label,
            label2id=label2id, cache_dir="cache",
            ignore_mismatched_sizes=True)  # self.pretrained_model = torch.load(  #     "b4_512x512_ade.pt") # for backbone

    def compute_metrics(self, eval_pred):

        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(logits_tensor, size=labels.shape[-2:], mode='bilinear',
                                                      align_corners=False, ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = self.metric._compute(predictions=pred_labels, references=labels, num_labels=len(self.id2label),
                                           ignore_index=False, reduce_labels=self.feature_extractor.do_reduce_labels, )

            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
            metrics.update({f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)})

            return metrics

    def train(self):
        self.preprocessing()
        self.trainer.train()

    def test(self):
        # Evaluate the model on the test dataset
        results = self.trainer.evaluate(eval_dataset=self.test_ds)

        print(results)


if __name__ == "__main__":
    print("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch Version:", torch.__version__)
    root_dir = "segmentation_dataset"
    segmentation = Segmentation(root_dir, number_of_labels=6, batch_size=4, epoch_number=20)
    segmentation.train()
    segmentation.test()
