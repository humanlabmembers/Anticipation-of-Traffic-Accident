import os
import cv2
import torch
from transformers import AutoImageProcessor, DetaForObjectDetection
from PIL import Image
import numpy as np
import time


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


traffic_labels = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]


deta_model_name = "jozhang97/deta-swin-large"
deta_processor = AutoImageProcessor.from_pretrained(deta_model_name)
deta_model = DetaForObjectDetection.from_pretrained(deta_model_name).to(device)

def checkoutpath(output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def nms(detections, iou_threshold=0.7):
   
    if len(detections) == 0:
        return []

    boxes = np.array([d['box'] for d in detections])
    scores = np.array([d['score'] for d in detections])
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        ious = iou(current_box, other_boxes)
        indices = indices[1:][ious < iou_threshold]
    
    return [detections[i] for i in keep]

def iou(box, boxes):
    
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])
    inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    box_area = (x2 - x1) * (y2 - y1)
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / union_area

def process_rider_and_motorcycle(detections):
    
 
    
    new_detections = []
    used_indices = set()

    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue
        box1 = det1['box']
        class1 = det1['class']

        if class1 not in [2, 3]:
            new_detections.append(det1)
            continue

        for j, det2 in enumerate(detections):
            if j <= i or j in used_indices:
                continue
            box2 = det2['box']
            class2 = det2['class']

            if class2 not in [2, 3] or class1 == class2:
                continue

            if is_overlapping(box1, box2):
               
                center1_x = (box1[0] + box1[2]) / 2
                center2_x = (box2[0] + box2[2]) / 2
               
                min_box_center_x = min(center1_x, center2_x)
                box_width = min(box1[2] - box1[0], box2[2] - box2[0])

                if min_box_center_x - 0.5 * box_width <= max(center1_x, center2_x) <= min_box_center_x + 0.5 * box_width:
                    merged_box = merge_boxes(box1, box2)
                    new_detections.append({'box': merged_box, 'score': max(det1['score'], det2['score']), 'class': 4})
                    used_indices.add(i)
                    used_indices.add(j)
                    break
        else:
            new_detections.append(det1)

    return new_detections

def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    new_x_min = min(x1_min, x2_min)
    new_y_min = min(y1_min, y2_min)
    new_x_max = max(x1_max, x2_max)
    new_y_max = max(y1_max, y2_max)

    return [new_x_min, new_y_min, new_x_max, new_y_max]


def extract(input_folder_path, output_folder_path, annotated_video_output_folder_path):
    for video_file in os.listdir(input_folder_path):
        if not video_file.endswith('.MP4'):
            continue
        output_file_path = os.path.join(output_folder_path, video_file.replace('.MP4', '.npz'))

        video_path = os.path.join(input_folder_path, video_file)
        frames = extract_frames(video_path)

        detections = []

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        annotated_video_path = os.path.join(annotated_video_output_folder_path, video_file)
        out = cv2.VideoWriter(annotated_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
        starttime= time.time()
        for i, frame in enumerate(frames):
            pil_image = Image.fromarray(frame)
            inputs = deta_processor(images=pil_image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = deta_model(**inputs)

            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = deta_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

            frame_detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = deta_model.config.id2label[label.item()]
                if label_name not in traffic_labels:
                    continue
                box = [int(i) for i in box]

                if label_name in ["truck", "bus", "car"]:
                    label_class = 1
                elif label_name in ["bicycle", "motorcycle"]:
                    label_class = 2
                elif label_name in ["person"]:
                    label_class = 3

                frame_detections.append({
                    'box': box,
                    'score': round(score.item(), 2),
                    'class': label_class
                })


            frame_detections = nms(frame_detections, iou_threshold=0.7)


            frame_detections = process_rider_and_motorcycle(frame_detections)


            if len(frame_detections) > 19:
                frame_detections = frame_detections[:19]
            while len(frame_detections) < 19:
                frame_detections.append({
                    'box': [0, 0, 0, 0],
                    'score': 0.0,
                    'class': 0
                })

            detections.append(frame_detections)
            

            for det in frame_detections:
                box = det['box']
                label_class = det['class']
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                label_text = f"Class: {label_class}, Score: {det['score']}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        end_time = time.time()
        print(end_time - starttime)
        cap.release()
        out.release()

        folder_name = os.path.basename(input_folder_path)

        if 'positive' in folder_name:
            label = 1
        elif 'negative' in folder_name:
            label = 0
        else:
            raise ValueError("The folder name is neither 'positive' nor 'negative'")

        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).cpu().numpy()


        detections_array = np.array([[det['box'] + [det['class']] for det in frame_detections] for frame_detections in detections])

        output_data = {
            'det': detections_array,
            'id': video_file,
            'label': label
        }

        
        
        np.savez(output_file_path, **output_data)
        print(video_file, 'saved')

    print("Processing completed. Results saved to:", output_folder_path)
    print("Annotated videos saved to:", annotated_video_output_folder_path)

if __name__ == '__main__':

    for videos in ['training/negative', 'training/positive', 'testing/negative', 'testing/positive']:
    # for videos in ['training/positive', 'testing/positive']:
        input_folder_path = '' + videos
        output_folder_path = '' + videos
        annotated_video_output_folder_path = '' + videos
        
        
        checkoutpath(output_folder_path)
        checkoutpath(annotated_video_output_folder_path)
        extract(input_folder_path, output_folder_path, annotated_video_output_folder_path)
