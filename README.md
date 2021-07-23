# Single human figure pose estimation from COCO

Use a simpler version of network in "[Learning to Shadow Hand-drawn Sketches](https://github.com/qyzdao/ShadeSketch)" to train single human pose estimation.

## Step 1 - data preprocessing
Install [cocoapi](https://github.com/cocodataset/cocoapi) to "./"

Download the coco dataset to "./annotations" and "./images" by runing this [.sh](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)

Move `coco_process.py` to "cocoapi/PythonAPI". This script 1) pull out human figures from coco dataset and make one single human figure per image. 
Human figure are centered in the canvas and have a white frame. Meanwhile, the keypoints are processed with regard to the new human figures. 
2) Create confidence map from keypoints. The processed dataset is written as `coco.npy` (recommend using >128G RAM to prepare data or spliting data to multiple .npy).

Examples:

<img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/131084_0.png" width="150"><img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/131084_0_c.png" width="150">.   <img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/262148_0.png" width="150"><img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/262148_0_c.png" width="150">


## Step 2 - training
Run `model.py`

After 3 epochs, start to converge. Evaluation results:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    Input images    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Target      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Prediction

<img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/289_img_GT.png" width="150"> <img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/289_heatmap_GT.png" width="150"> <img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/289_heatmap.png" width="150">

<img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/291_img_GT.png" width="150"> <img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/291_heatmap_GT.png" width="150"> <img src="https://github.com/qyzdao/single-human-pose-estimation/blob/main/imgs/291_heatmap.png" width="150">
