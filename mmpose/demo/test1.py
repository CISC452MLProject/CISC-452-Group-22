from mmpose.apis import MMPoseInferencer

img_path = '../../img/frame0.jpg'

inferencer = MMPoseInferencer('human')

result_gen = inferencer(img_path, show=True)
result = next(result_gen)
