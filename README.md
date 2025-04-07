# KD-RSCC: A Karras Diffusion Framework for Efficient Remote Sensing Change Captioning
> __KD-RSCC: A Karras Diffusion Framework for Efficient Remote Sensing Change Captioning__  
> Xiaofei Yu, Jie Ma*，Liqiang Qiao

##  Model Architecture
The proposed KD-RSCC consists of:
- (A) the Karras diffusion framework with sampling method
- (B) image difference-conditioned denoiser via $I_{\text{diff}}$
  - (1) Change Feature Extraction
  - (2) Semantic-Spatial Alignment
  - (3) Hierarchical Self-Attention Refinement.

![method](https://github.com/user-attachments/assets/0ba213d5-a4a6-4b0a-a8a0-30e4594aa72a)

### Datasets
#### LEVIR-CC
- A large-scale RSICC dataset with 10,077 bi-temporal image pairs and 50,385 captions.
- Covers multiple semantic change types: buildings, roads, vegetation, parking lots, water.
- Resized images: 256×256.

Download Source:
-Thanks for the Dataset by Liu et. al:[[GitHub](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)].
Put the content of downloaded dataset under the folder 'data'
```python
path to ./data:
                ├─LevirCCcaptions.json
                ├─images
                  ├─train
                  │  ├─A
                  │  ├─B
                  ├─val
                  │  ├─A
                  │  ├─B
                  ├─test
                  │  ├─A
                  │  ├─B
```


## Installation and Dependencies
```python
git clone git@github.com:Fay-Y/KD_RSCC.git
cd KD_RSCC
conda create -n KD_env python=3.8
conda activate KD_env
pip install -r requirements.txt
```
## Preparation
Preprocess the raw captions and image pairs:
```python
python word_encode.py
python img_preprocess.py
```

## Training
 To train the proposed Diffusion-RSCC, run the following command:
```python
sh train.sh
```


## Prediction samples
Prediction results in test set with 5 Ground Truth captions are partly shown below, proving the effectiveness of our model. 
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/eaf7ba0c-1a4d-44cd-9d11-84bfda0058ab" alt="compare2" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/b61bad59-afd0-4313-9b97-d7ab859222eb" alt="compare1" width="500"/></td>
  </tr>
</table>

## TODO
- [ ] Release training logs and checkpoints
- [ ] Support more RSICC datasets






