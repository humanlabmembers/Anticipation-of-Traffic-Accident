from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

def extract_object_features_from_video(input_path, output_path):
    # 定义全连接层，将4096维特征降到512维
    fc_layer = nn.Linear(4096, 512, bias=False).to(torch.device('cuda'))

    for video_file in os.listdir(input_path):
        if not video_file.endswith('.npz'):
            continue

        input_file_path = os.path.join(input_path, video_file)
        output_file_path = os.path.join(output_path, video_file)
        if os.path.exists(output_file_path):
            continue

        data = np.load(input_file_path, allow_pickle=True)
        new_data = {}

        for array_name in data.files:
            array = data[array_name]
            if array.shape[-1] == 4096:
                # 将数据转换为Tensor，并移动到GPU
                tensor = torch.from_numpy(array).float().to(torch.device('cuda'))
                # 通过全连接层降维
                reduced_tensor = fc_layer(tensor).cpu().detach().numpy()
                new_data[array_name] = reduced_tensor
                # print(f"Array '{array_name}' shape changed from {array.shape} to {reduced_tensor.shape}")
            else:
                new_data[array_name] = array

        # 保存转换后的数据
        np.savez(output_file_path, **new_data)
        # print(f"Processed and saved: {output_file_path}")

if __name__ == '__main__':
    base_input_path = ''
    base_output_path = ''

    for videos in ['training/negative', 'training/positive', 'testing/negative', 'testing/positive']:
        in_path = os.path.join(base_input_path, videos)
        out_path = os.path.join(base_output_path, videos)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        extract_object_features_from_video(
            input_path=in_path,
            output_path=out_path
        )
