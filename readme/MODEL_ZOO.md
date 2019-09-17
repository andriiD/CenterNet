# MODEL ZOO

### Common settings and notes

- The experiments are run with pytorch 1.2, CUDA 10.1, and CUDNN 7.1.
- Training times are measured on our servers with 8 TITAN V GPUs (12 GB Memeory).
- Testing times are measured on our local machine with TITAN Xp GPU. 
- The models can be downloaded directly from [Google drive]().

## Object Detection


### COCO

| Model                    | GPUs |Train time(h)| Test time (ms) |   AP               |  Download | 
|--------------------------|------|-------------|----------------|--------------------|-----------|
|[ctdet\_coco\_dla\_1x](../experiments/ctdet_coco_dla_1x.sh)  |   8  | 57          |  19 / 36 / 248 | 36.3 / 38.2 / 40.7 | [model](https://drive.google.com/open?id=1r89_KNXyDyvUp8NggduG9uKQTMU2DsK_) |
|[ctdet\_coco\_dla\_2x](../experiments/ctdet_coco_dla_2x.sh)  |   8  | 92          |  19 / 36 / 248 | 37.4 / 39.2 / 41.7 | [model](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT) |
|[ctdet\_coco\_resdcn101](../experiments/ctdet_coco_resdcn101.sh)|   8  | 65          |  22 / 40 / 259 | 34.6 / 36.2 / 39.3 | [model](https://drive.google.com/open?id=1bTJCbAc1szA9lWU-fvVw52lqR3U2TTry) |
|[ctdet\_coco\_resdcn18](../experiments/ctdet_coco_resdcn18.sh) |   4  | 28          |  7 / 14 / 81   | 28.1 / 30.0 / 33.2 | [model](https://drive.google.com/open?id=1b-_sjq1Pe_dVxt5SeFmoadMfiPTPZqpz) |

#### Notes

- All models are trained on COCO train 2017 and evaluated on val 2017. 
- We show test time and AP with no augmentation / flip augmentation / multi scale (0.5, 0.75, 1, 1.25, 1.5) augmentation. 
- Results on COCO test-dev can be found in the paper or add `--trainval` for `test.py`. 
- exdet is our re-implementation of [ExtremeNet](https://github.com/xingyizhou/ExtremeNet). The testing does not include edge aggregation.
- For dla and resnets, `1x` means the training schedule that train 140 epochs with learning rate dropped 10 times at the 90 and 120 epoch (following [SimpleBaseline](https://github.com/Microsoft/human-pose-estimation.pytorch)). `2x` means train 230 epochs with learning rate dropped 10 times at the 180 and 210 epoch. The training schedules are **not** carefully investigated.
- The hourglass trained schedule follows [ExtremeNet](https://github.com/xingyizhou/ExtremeNet): trains 50 epochs (approximately 250000 iterations in batch size 24) and drops learning rate at the 40 epoch.
- Testing time include network forwarding time, decoding time, and nms time (for ExtremeNet).
- We observed up to 0.4 AP performance jitter due to randomness in training. 


## Human pose estimation

### COCO

| Model                    | GPUs |Train time(h)| Test time (ms) |   AP        |  Download | 
|--------------------------|------|-------------|----------------|-------------|-----------|
|[multi\_pose\_dla_1x](../experiments/multi_pose_dla_1x.sh)   |   8  |30           | 44             | 54.7        | [model](https://drive.google.com/open?id=1VeiRtuXfCbmhQNGV-XWL6elUzpuWN-4K) |
|[multi\_pose\_dla_3x](../experiments/multi_pose_dla_3x.sh)   |   8  |70           | 44             | 58.9        | [model](https://drive.google.com/open?id=1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI) |

#### Notes
- All models are trained on keypoint train 2017 images which contains at least one human with keypoint annotations (64115 images).
- The evaluation is done on COCO keypoint val 2017 (5000 images).
- Flip test is used by default.
- The models are fine-tuned from the corresponding center point detection models.
- Dla training schedule: `1x`: train for 140 epochs with learning rate dropped 10 times at the 90 and 120 epoch.`3x`: train for 320 epochs with learning rate dropped 10 times at the 270 and 300 epoch.
- Hourglass training schedule: `1x`: train for 50 epochs with learning rate dropped 10 times at the 40 epoch.`3x`: train for 150 epochs with learning rate dropped 10 times at the 130 epoch.
