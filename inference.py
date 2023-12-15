import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation

import scale_bar


class Model:
    def __init__(self, model_path, colormap_path, output_path=None):
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()  # Set the model to evaluation mode
        self.feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
        self.recording_path = output_path
        self.color_map_path = colormap_path
        self.id2color = None
        self.id2name = None
        self.name2id = None

    def preprocessing(self, image):
        # prepare the image for the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoding = self.feature_extractor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(device)

        return pixel_values

    def forward_pass(self, pixel_values):
        # forward pass
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()

        return logits

    def get_image_shape(self, image):
        height, width, channels = image.shape

        return height, width, channels

    def merge_original_image_width_segmentation_outputs(self, image, segmentation_result):
        # Show image + mask
        image = np.array(image) * 0.5 + segmentation_result * 0.5
        image = image.astype(np.uint8)

        return image

    def read_color_map(self):
        self.color_map = pd.read_csv(self.color_map_path, sep="\t", header=None)
        self.color_map.columns = ["Class", "R", "G", "B", "ID"]
        id2color = {id: [R, G, B] for id, (R, G, B) in
                    enumerate(zip(self.color_map.R, self.color_map.G, self.color_map.B))}
        id2name = {}
        for _, row in self.color_map.iterrows():
            class_label = row['ID'] - 1
            rgb_values = row['Class']
            id2name[class_label] = rgb_values
        name2id = {key: value for (value, key) in id2name.items()}
        return id2color, id2name, name2id

    def predict(self, image):
        preprocessed_image = self.preprocessing(image)
        logits = self.forward_pass(preprocessed_image)
        height, width, _ = image.shape
        upsampled_logits = torch.nn.functional.interpolate(logits,
                                                        size=(height,width),
                                                        mode="bilinear",
                                                        align_corners=False)

        seg = upsampled_logits.argmax(dim=1)[0].cpu()
        color_seg =  np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        if self.id2color is None:
            self.id2color, self.id2name, self.name2id = self.read_color_map()

        for label, color in self.id2color.items():
            if not label == 0:
                color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        output_image = self.merge_original_image_width_segmentation_outputs(image, color_seg)
        return output_image, seg

    def inference_image(self, image_path):
        images = glob.glob(image_path)
        counter = 0
        for image_path in images:
            im_name = os.path.basename(image_path)
            image = cv2.imread(image_path)
            output_image, mask = self.predict(image)
            cv2.imwrite(f"{self.recording_path}/{im_name}", output_image)
            for label, name in self.id2name.items():
                pixel_meter_ratio = scale_bar.find_length_scale_bar_web(image, int(im_name[:-5]))
                area = scale_bar.calc_mask_area(mask.numpy(), label, pixel_meter_ratio)
                print(f'{name} : {area} meter squared')
            counter += 1


if __name__ == "__main__":
    model_path = "model_outdir/checkpoint-610"
    color_map_path = "segmentation_dataset/colormap.txt"  # Path of the colormap.txt (probably in datasets)
    output_path = 'outputs'
    image_path = "scale_inputs/*.png"  # The image-images path for want to inference (RegEx)

    model = Model(model_path, color_map_path, output_path=output_path)
    model.inference_image(image_path)
