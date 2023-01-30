# 뇌경색 판독문 자동 생성 - img2txt

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
3. CT data (뇌출혈 데이터)
  * CC_case (3,222개) : 춘천성심병원 CT 데이터, Dicom파일
  * HL_case (7,307개) : 평촌성심병원 CT 데이터, Dicom파일
  * HL_normal (30,576개) : 평촌성심병원 CT normal 데이터, Dicom파일
    - CC_case, HL_case, HL_normal 모두 Annotation 파일 없음(23.01.31 기준)
  * hallym_CT (ver 1.0 2022.3.27).xlsx : 평촌성심병원 CT 판독문
  * ich_chuncheon.xlsx : 춘천성심병원 CT 판독문
  * normal_chuncheon.xlsx :  평촌성심병원 CT normal 판독문
    - 판독문 3개 모두 CT데이터와 id가 일부 match되지 않음 (23.01.31 기준) => 김철호 교수님께 문의

## Train the model
data_type = ct / mri 두가지 중 하나로 설정
data_channel = mri일 경우 3으로 설정, ct일 경우 원하는 채널 수 만큼 설정
```
python train_visualGPT.py  --exp_name visualGPT \
                       --train_data_path /home/lab/sangjee/strok/data/ctdata_train.csv \
                       --test_data_path /home/lab/sangjee/strok/data/ctdata_test.csv \
                       --val_data_path /home/lab/sangjee/strok/data/ctdata_val.csv \
                       --epoch 100 \
                       --patience 5 \
                       --batch_size 16 \
                       --eval_batch_size 16 \
                       --num_workers 4 \
                       --head 12 \
                       --logs_folder /home/lab/sangjee/strok/tensorlog \
                       --random_seed 42 \
                       --gpt_model_type gpt \
                       --lr 1e-4 \
                       --log_file /home/lab/sangjee/strok/log/visualGPT.txt \
                       --gradient_accumulation_steps 1 \
                       --num_train_epochs 3.0 \
                       --optimizer_type adamw \
                       --max_grad_norm 1.0 \
                       --train_percentage 1.0 \
                       --reinforcement_lr 1e-5 \
                       --decoder_layer 12 \
                       --encoder_layer 3 \
                       --tau 0.0 \
                       --data_type ct \
                       --data_channel 20
```


## VisualGPT

Our Paper [VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning](https://arxiv.org/abs/2102.10407)

### Main Architecture of Our VisualGPT
![image](images/final_architecture.jpg)


### Download the GPT-2 pretrained weights
```
curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
```

### Data preparation
We provide the COCO dataset for downloading. Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it.
and [coco_detections.hdf5](https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx), in which the data is stored in a `<key, value>` where key is the image id and value is a tensor (N, 2048). N it the number of detections


## Acknowledgement
This code used resources from [Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer) and [Transformers](https://github.com/huggingface/transformers)