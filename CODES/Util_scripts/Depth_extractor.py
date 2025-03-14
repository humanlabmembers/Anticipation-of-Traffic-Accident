import numpy as np
import torch
from torchvision import transforms
import cv2
from ZoeDepth.zoedepth.utils.misc import colorize, save_raw_16bit
import os
import re
import torch.nn.functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_zoe = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True, trust_repo=True)
model_zoe.to(device)

def process_and_compress_video(video_path, save_path):
    video_capture = cv2.VideoCapture(video_path)
    video_id = os.path.basename(video_path).split('.')[0]
    
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return

    depth = []
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  


        
        tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth_frame = model_zoe.infer_pil(tensor)  
        depth_frame = torch.from_numpy(depth_frame)
    
        depth.append(depth_frame)

    video_capture.release()

    depth_id = np.stack(depth, axis=0)
    
    
    depth_tensor = torch.tensor(depth_id).unsqueeze(1).to(device)  

    target_size = (72, 128)


    compressed_depth_tensor = F.interpolate(depth_tensor, size=target_size, mode='bilinear', align_corners=False)

  
    compressed_depth_tensor = compressed_depth_tensor.squeeze(1)


    compressed_depth_data = compressed_depth_tensor.cpu().numpy()


    compressed_save_path = os.path.join(save_path, f'WM_{video_id}.npz')
    np.savez_compressed(compressed_save_path, depth_data=compressed_depth_data)
    print(f"Processed and saved: {compressed_save_path}")

def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)
    if match:
        return match.group(0)
    return None

def process_videos(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.MP4'):
            video_path = os.path.join(input_folder, file_name)
            process_and_compress_video(video_path, output_folder)

if __name__ == '__main__':
    base_input_folder = ''
    base_output_folder = ''
    # for videos in ['training/negative', 'training/positive', 'testing/negative', 'testing/positive']:
    #     input_folder_path = os.path.join(base_input_folder, videos)
    #     output_folder_path = os.path.join(base_output_folder, videos)
    
    process_videos(base_input_folder, base_output_folder)
