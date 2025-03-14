from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import cv2
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

# input_txt= ''

def count_labels(input_file):
    label_counts = {}
    error_lines = []
    pattern = re.compile(r"(\d+),(\[.*?\]),")

    with open(input_file, 'r') as file:
        for line_number, line in enumerate(file, 1):
            match = pattern.match(line.strip())
            if match:
                id = match.group(1)
                labels_str = match.group(2)

                try:
                    labels = eval(labels_str)
                    count_zeros = labels.count(0)
                    label_counts[id] = count_zeros
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing labels on line {line_number}: {e}")
                    error_lines.append(line)
            else:
                print(f"Error matching pattern on line {line_number}")
                error_lines.append(line)

    return label_counts

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.dim_feat = 4096
        

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output


def extract_object_features_from_video(video_path, object_detection_npz_path, output_path):
    for video_file in os.listdir(video_path):



        if not video_file.endswith('.MP4'):
            continue
        object_detection_npz = os.path.join(object_detection_npz_path, video_file.replace('.MP4', '.npz'))
        output_file_path = os.path.join(output_path, 'WM_'+video_file.replace('.MP4', '.npz'))


        device = "cuda:0"
        extractor = VGG16().to(device)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


        id = video_file[:-4]
        folder_name = os.path.basename(video_path)

        if folder_name == 'positive':
            label = 1
        elif folder_name == 'negative':
            label = 0
        else:
            raise ValueError("The folder name is neither 'positive' nor 'negative'")
        
        if label == 1:
            toa=90
        else:
            toa=101


        

        video = os.path.join(video_path, video_file)
        video_capture = cv2.VideoCapture(video)

        object_detection_data = np.load(object_detection_npz, allow_pickle=True)
        detections = object_detection_data['det']


        if not os.path.exists(output_path):
            os.makedirs(output_path)

        num_frames, num_objects, _ = detections.shape
        features_array = np.zeros((num_frames, num_objects, 4096))  
        num_frames=100
        ffeatures= np.zeros((num_frames,  4096))   
        frame_id = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret or frame_id >= num_frames:
                break

            frame_objects = detections[frame_id]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            transformed_frame = transform(frame_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                ffeat = extractor(transformed_frame)
                ffeat =ffeat.cpu().numpy().squeeze()
                ffeatures[frame_id,  :] = ffeat
            for obj_id, obj in enumerate(frame_objects):
                xmin, ymin, xmax, ymax = map(int, obj[:4])
                cropped_image = frame_rgb[ymin:ymax, xmin:xmax]
                if cropped_image.size == 0:
                    continue
                cropped_pil = Image.fromarray(cropped_image)
                transformed_image = transform(cropped_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = extractor(transformed_image)
                    features = features.cpu().numpy().squeeze()
                features_array[frame_id, obj_id, :] = features

            frame_id += 1


        np.savez(output_file_path, ffeat=ffeatures, det=detections, features=features_array, toa=toa,label=label,ID=id)
        print(f'{video_file} finished')
        video_capture.release()


if __name__ == '__main__':
    for videos in ['training/positive','training/negative','testing/negative', 'testing/positive']:
        video_path = '' + videos
        detection_path = '' + videos
        out_path = '' + videos

        extract_object_features_from_video(
            video_path=video_path,
            object_detection_npz_path=detection_path,
            output_path=out_path
        )
