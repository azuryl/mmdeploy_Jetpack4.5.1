## MMOCR Support

MMOCR is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMOCR installation tutorial

Please refer to [install.md](https://github.com/open-mmlab/mmocr/blob/main/docs/install.md) for installation.

### List of MMOCR models supported by MMDeploy

|    model     |       task       | OnnxRuntime |    TensorRT   | NCNN |  PPL  | model config file(example)                                                                |
| :----------  | :--------------: | :---------: | :-----------: | :---:| :---: | :---------------------------------------------------------------------------------------  |
| DBNet        | text-detection   |      Y      |       Y       |   Y  |   Y   | $PATH_TO_MMOCR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py                    |
| CRNN         | text-recognition |      Y      |       Y       |   Y  |   N   | $PATH_TO_MMOCR/configs/textrecog/crnn/crnn_academic_dataset.py                            |
| SAR          | text-recognition |      Y      |       N       |   N  |   N   | $PATH_TO_MMOCR/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py                 |

### Reminder

None

### FAQs

None