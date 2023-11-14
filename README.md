## Solution for dacl-challenge: Semantic Bridge Damage Segmentation

### How to reproduce the result

#### 1. Environment

1 RTX 4090 24GB VRAM
CUDA 11.6

Install packages:
```
conda install -f environment.yml
conda activate dacl
```

### 2. Prepare dataset

Download all dataset from [link](https://eval.ai/web/challenges/challenge-page/2130/overview)

run this file to generate mask
```
python prepare_data.py
python convert_to_gray.py
mkdir -p dataset/images/all
mkdir -p dataset/masks/all
cp -r dataset/images/train/* dataset/images/all
cp -r dataset/images/validation/* dataset/images/all

cp -r dataset/masks/train/* dataset/masks/all
cp -r dataset/masks/validation/* dataset/masks/all

```

Plese see dataset.txt for more detail

### 3. Train

```
cd mmsegmentation
pip install -e .
bash scripts/train.sh
```

After training done, the output of model will be shown in work_dir: **upernet_convnext_base_fp16_640x640_120k_all/iter_120000.pth**

### 4. Inference

#### 4.1 For testdev

We will submit 1012 images to eval.ai. Run this to infer:

```
python infer_mmseg_dev.py
python mask2json_dev.py
python json2jsonl_dev.py
```

output of testdev named *dev.jsonl* will be appeared in *output/dev/jsonl_output*

Submit to eva.ai [link](https://eval.ai/web/challenges/challenge-page/2130/submission) into tab Development Phase and check the result

#### 4.2 For testchallenge 

We will submit 1012 images from testdev and 998 images from testchallenge. Run this to infer:

```
python infer_mmseg_dev.py
python infer_mmseg_test.py
python json2jsonl_dev.py
python json2jsonl_test.py
cp -r output/dev/json_output/* output/test/json_output
python json2jsonl_test.py

```
output of testchallenge named *test.jsonl* will be appeared in *output/test/jsonl_output*

Submit to eva.ai [link](https://eval.ai/web/challenges/challenge-page/2130/submission) into tab Testfinal Phase and check the result

### 5. References

#### 1. [Eval.ai](https://eval.ai/web/challenges/challenge-page/2130)
#### 2. [1st Workshop on Vision-based Structural Inspections in Civil Engineering](https://dacl.ai/workshop.html)
#### 3. [dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit)
#### 4. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
#### 5. [dacl10k](https://arxiv.org/abs/2309.00460)

