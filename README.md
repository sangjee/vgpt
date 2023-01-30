# 뇌경색 판독문 자동 생성 - img2txt

## 진행 내용
1. 판독문 data review
   - 대부분의 판독문이 질병에 관한 정보 / 위치 정보의 형태로 이루어져 있음
   - 오타 존재함
   - 약어 사용
   - 용어 통일 필요함
   - 한국어와 영어가 혼용된 판독문 일부 존재함
   - 판독문 하나에 여러가지 사항이 나열되어 있음 (slice1개당 1개의 판독문이 매치가 안됨)
2. CLIP 모델 확인
    - hallym data 약 1000개 정도 사용
    - Image Encoder : Resnet50
    - Text Encoder : BioBERT
    - BLUE_1 score 기준으로 성능 확인 : 약 0.3 정도의 성능을 보임
    - 일반화 성능이 좋지 않음
    - 동일한 판독문에 대해 상이한 영상 슬라이드로 학습 시, 학습되지 않음
    - 단층 영상에서 1개의 슬라이스 만을 사용하여 학습 시, 학습이 되지 않음 => 3개 이상 사용하면 학습됨
3. BERT fine-tuning
    - Brain MRI report.xls 판독문 모음을 이용하여 뇌경색 data에 맞는 BERT를 학습시켜 확인
    - Masked Language Model(MLM) 사용
    - CLIP의 text Encoder를 fine-tuning한 BERT로 바꿔도 0.001 정도의 성능 향상이 있음 => 거의 효과 없음
4. Image Encoder를 2d에서 3d로 변경
    - 2D Resnet50 : BLUE1 0.349
    - 3D Resnet50 : BLUE1 0.0.317
    - 여전히 일반화 성능이 좋지 않음
    - 오히려 성능이 조금 떨어짐
    - CLIP의 경우 이미지 전체와 텍스트 전체의 유사도를 구하는 모델로 생성모델과 차이가 있음
5. Review - Image Captioning Model
    - SimVML(2021.08)
    - BLIP(2022.02)
    - CoCa(2022.05)
    - GRIT(2022.07)
6. Review - Medical Img2txt Model
    - MedViLL(2021.05)
    - MedCLIP(2021.10)
    - VisualGPT(2022.05)
7. Review - Transformers in Medical Image Analysis
8. Open data 확인
    - Image Captioning Data
      - ROCO(Radiology Objects in Context) : Radiology data(Computer Tomography, Ultrasound, X-Ray, Fluoroscopy, Positron Emission Tomography, Mammography, Magnetic Resonance Imaging, Angiography.)
      - IU Chest X-ray
    - Image MRI Data
      - NYU fastMRI
      - BraTS2021
      - Gazi Brains
      - MRBrainS18
      - iSeg
      - IXI
9. visualGPT 적용
    1. visualGPT Review
      - Encoder-decoder 구조
      - 적은 양의 학습 데이터를 가지고 성능을 달성함
      - IU x-ray data의 SOTA 성능을 달성함
    2. gpt2 - pretraining with hallym data
      - 한림대 vocab을 사용하여 pretraining
      - hallym data 판독문과 embedding용으로 제공해준 판독문을 사용하여 pretraining
      - Hallym data 고유 단어 : 456
      - Embedding용 고유 단어 : 3968  => hallym data와 중복되는 단어 293개
      - Hallym data : vocab size = 1,255
      - 기존 gpt2가 성능이 더 좋음
    3. masked 이미지 사용
      - Mask 기준 영역만 추출하여 사용
      - 20개의 MRI 단층 영상 사용
      - Mask 크기 기준 top 20개 사용
      - 0.03~0.04의 성능 향상이 있음
    4. CT data 사용


## Data
### DWI 뇌경색 데이터
1. hallym data
  * disease (1,200개) : 뇌경색 dwi data, Dicom파일
  * normal (100개) : dwi data, Dicom파일
  * mr_txt.xlsx : disease/normal 판독문 및 환자 정보
  * Brain MRI report.xls : embedding용 mri 판독문 모음
  * 데이터중심병원 : 3520개 => 판독문 및 anotation 파일 없음(23.01.31 기준)
2. knu data
  * disease (1,182개) : Dicom파일
  * normal (22개) : Dicom파일
  * 강원대학교 병원.xlsx : Tabular data
  * 영상 판독지 요청 자료.xls : disease/normal 판독문 및 환자 정보 => 정제되지 않음, 검토 필요한 데이터
### CT 뇌출혈 데이터
1. CT data
  * CC_case (3,222개) : 춘천성심병원 CT 데이터, Dicom파일
  * HL_case (7,307개) : 평촌성심병원 CT 데이터, Dicom파일
  * HL_normal (30,576개) : 평촌성심병원 CT normal 데이터, Dicom파일
    - CC_case, HL_case, HL_normal 모두 Annotation 파일 없음(23.01.31 기준)
  * hallym_CT (ver 1.0 2022.3.27).xlsx : 평촌성심병원 CT 판독문
  * ich_chuncheon.xlsx : 춘천성심병원 CT 판독문
  * normal_chuncheon.xlsx :  평촌성심병원 CT normal 판독문
    - 판독문 3개 모두 CT데이터와 id가 일부 match되지 않음 (23.01.31 기준) => 김철호 교수님께 문의
    - case 판독문의 경우 ich(뇌출혈)=1인 데이터 사용
    - nomral 판독문의 경우 ich(뇌출혈)=0인 데이터 사용
    - series=1, new_series=1인 데이터 사용 => series=ct찍은 순서(여러 번 찍은 환자도 있기 때문)
    - id = ct data 폴더명과 일치, 환자 번호

## Train the model

**visualgpt gitlab 코드를 참고하여 데이터 부분만 변경하여 사용, 아래 gitlab주소 활용할 것**

**[주요 추가,변경 부분]**
- CustomDataModule.py
- CustomDataset.py
  - mri 데이터의 경우 mask 기준 가장 병변이 많은 slice 3개를 dicom 파일에서 뽑아 사용, dicom을 hdf5로 변환하여 사용
  - CT데이터의 경우 dicom 파일을 nifti파일로 변환하여 사용

**[train data 활용 예시 - mri]**
|caption|image|input_img1|input_img2|input_img3|
|------|---|---|---|---|
|focal small diffusion restriction|/data/hdf5_hallym/train/HALLYM_CC_0391.hdf5|input_00000010|input_00000029|input_00000016|
|multiple diffusion restriction|/data/hdf5_hallym/train/HALLYM_CC_0214.hdf5|input_00000010|input_00000011|input_00000014|

**[train data 활용 예시 - ct]**
|caption|image|
|------|---|
|No demonstrable abnormal finding.|/data/nifti/HL_case/104634/104634.nii.gz|
|Left basal ganglia intracerebral hemorrhage|/data/nifti/CC_case/2085/2085.nii.gz|

**[train]**

* data_type = ct / mri 두가지 중 하나로 설정
* data_channel = mri일 경우 3으로 설정, ct일 경우 원하는 채널 수 만큼 설정

```
python train_visualGPT.py  --exp_name visualGPT
  --train_data_path /data/mridata_train.csv
  --test_data_path /data/mridata_test.csv
  --val_data_path /data/mridata_val.csv
  --epoch 100
  --patience 5
  --batch_size 16
  --eval_batch_size 16
  --num_workers 4
  --head 12
  --logs_folder /home/lab/sangjee/strok/tensorlog
  --random_seed 42
  --gpt_model_type gpt
  --lr 1e-4
  --log_file /home/lab/sangjee/strok/log/visualGPT.txt
  --gradient_accumulation_steps 1
  --num_train_epochs 3.0
  --optimizer_type adamw
  --max_grad_norm 1.0
  --train_percentage 1.0
  --reinforcement_lr 1e-5
  --decoder_layer 12
  --encoder_layer 3
  --tau 0.0
  --data_type mri
  --data_channel 3
```


## VisualGPT
[gitlab](https://github.com/Vision-CAIR/VisualGPT)

Our Paper [VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning](https://arxiv.org/abs/2102.10407)

### Main Architecture of Our VisualGPT
![image](images/final_architecture.jpg)


### Download the GPT-2 pretrained weights
```
curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
```


## Acknowledgement
This code used resources from [Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer) and [Transformers](https://github.com/huggingface/transformers)