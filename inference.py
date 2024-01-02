import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation

import scale_bar
import solarEstimate


class Model:
    def __init__(self, model_path, colormap_path, output_path=None):
        self.color_map = None
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
        upsampled_logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear",
                                                           align_corners=False)

        seg = upsampled_logits.argmax(dim=1)[0].cpu()
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        if self.id2color is None:
            self.id2color, self.id2name, self.name2id = self.read_color_map()

        for label, color in self.id2color.items():
            if not label == 0:
                color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        output_image = self.merge_original_image_width_segmentation_outputs(image, color_seg)
        return output_image, seg

    def get_coordinates(self, image):
        # Function to be called when the mouse is clicked
        def onclick(event):
            ix, iy = event.xdata, event.ydata
            print(f'x = {ix}, y = {iy}')
            coords.append((int(ix), int(iy)))

            # If two points have been clicked, disconnect the mouse event
            if len(coords) == 2:
                plt.disconnect(cid)
                plt.close()

        # Display the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        coords = []

        # Connect the mouse click event to the onclick function
        cid = plt.connect('button_press_event', onclick)

        plt.show()
        return coords

    def inference_image(self, image_path, coordinate, month='Year', degree=27):
        images = glob.glob(image_path)
        counter = 0
        for image_path in images:
            im_name = os.path.basename(image_path)
            image = cv2.imread(image_path)
            pixel_meter_ratio = scale_bar.find_length_scale_bar_web(image, int(im_name[:-5]))
            coord1, coord2 = self.get_coordinates(image)
            x1, y1 = coord1
            x2, y2 = coord2
            roi = image[y1:y2, x1:x2]
            output_image, mask = self.predict(roi)
            cv2.imwrite(f"{self.recording_path}/{im_name}", output_image)
            area_Dict = {'N': 0, 'S': 0, 'W': 0, 'E': 0, 'F': 0}
            for label, name in self.id2name.items():
                area = scale_bar.calc_mask_area(mask.numpy(), label, pixel_meter_ratio)
                if name == 'flat':

                    area_Dict['F'] = area
                else:
                    area_Dict[name] = area
            print(area_Dict)
            print(solarEstimate.estimate(area_Dict, coordinate, month, degree))
            counter += 1


if __name__ == "__main__":
    model_path = "model_outdir/checkpoint-820"
    color_map_path = "segmentation_dataset/colormap.txt"  # Path of the colormap.txt (probably in datasets)
    output_path = 'outputs'
    image_path = "scale_inputs/5m.png"  # The image-images path for want to inference (RegEx)

    model = Model(model_path, color_map_path, output_path=output_path)
    coordinate = '40.2389x33.0328'  # lat x long format coordinate
    month = 'Year'  # if Annual 'Year', else name of the month
    degree = 30  # The angle between rooftop and surface
    model.inference_image(image_path, coordinate, month, degree)
