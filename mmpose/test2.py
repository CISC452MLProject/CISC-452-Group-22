import requests
import subprocess
import os
from mmpose.apis.inferencers import MMPoseInferencer

training = '../training_climbData.json'

dec_checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

dec_checkpoint = requests.get(dec_checkpoint_url).content



command = ["../Scripts\\python.exe", "tools/misc/generate_bbox_file.py", "demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py", dec_checkpoint_url,"bbox.json", "--pose-config", training ]
# mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-18_8xb32-210e_coco-384x288.py


subprocess.run(command)




