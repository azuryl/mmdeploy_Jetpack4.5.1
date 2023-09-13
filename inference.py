from mmdeploy.apis import inference_model
import time 

#static
#deploy_cfg = "/data/azuryl/mmdeploy_0.7.0/configs/mmdet/instance-seg/instance-seg_tensorrt_static-800x1344.py" 
deploy_cfg = "/data/azuryl/mmdeploy_0.7.0/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_static-800x1344.py"#0.2632/1000
#deploy_cfg = "/data/azuryl/mmdeploy_0.7.0/configs/mmdet/instance-seg/instance-seg_tensorrt-int8_static-800x1344.py"#0.627/2000 0.581/2w

#dynamic
#deploy_cfg = "/data/azuryl/mmdeploy_0.7.0/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py"#0.70 0.575/1000 0.575/2000 0.577/2w #0.557/2k #2.17FPS
#deploy_cfg = "/data/azuryl/mmdeploy_0.7.0/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py"#0.77 0.5749/2000  0.574/2w  #0.237/2k,0.233/2k 6.51FPS
#deploy_cfg = "/data/azuryl/mmdeploy_0.7.0/configs/mmdet/instance-seg/instance-seg_tensorrt-int8_dynamic-320x320-1344x1344.py"#0.577/1000 0.539/2w  0.216/1k,0.214/2k,0.211/2w 8.32FPS

model_cfg = "/data/azuryl/mmdetection_2.27.0/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py"

#backend_files = ["/data/azuryl/mmdeploy_model/maskrcnn_d320_1344/end2end.engine"]
#backend_files = ["/data/azuryl/mmdeploy_model/maskrcnn_f16_d320_1344/end2end.engine"]
#backend_files = ["/data/azuryl/mmdeploy_model/maskrcnn_int8_d320_1344/end2end.engine"]
backend_files = ["/data/azuryl/mmdeploy_model/maskrcnn_fs800_1344/end2end.engine"]
img = "/data/azuryl/mmdetection_2.27.0/demo/demo.jpg" 
device = 'cuda:0'

start = time.time()
result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
#print("time:",time.time() -start)
#print(result)
