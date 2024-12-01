import requests
import subprocess
import os
from mmpose.apis.inferencers import MMPoseInferencer

dec_checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
pos_config_url = "https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288-a3aef5c4_20210626.pth"

python_executable=os.path.realpath('\\'.join(__file__.split('\\')[:-3])+'\\Scripts\\python.exe')


pos_config = requests.get(pos_config_url).content
dec_checkpoint = requests.get(dec_checkpoint_url).content

# command = [python_executable, "demo/topdown_demo_with_mmdet.py", "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py", dec_checkpoint_url,
#            "configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
#             pos_config_url, "--input", "../cropped/34_2018-07-06_2.mp4", "--output-root=vis_results/demo", "--show", "--draw-heatmap" ]
# inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')
command = [python_executable, "demo/topdown_demo_with_mmdet.py", "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py", dec_checkpoint_url,
           "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-30_8xb32-210e_coco-384x288.py",
            pos_config_url, "--input", "../cropped/34_2018-07-06_2.mp4", "--output-root=vis_results/demo", "--show", "--draw-heatmap" ]
# mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-18_8xb32-210e_coco-384x288.py

print(command)
print(os.path.realpath('\\'.join(__file__.split('\\')[:-2])))
subprocess.run(command, cwd=os.path.realpath('\\'.join(__file__.split('\\')[:-2])))