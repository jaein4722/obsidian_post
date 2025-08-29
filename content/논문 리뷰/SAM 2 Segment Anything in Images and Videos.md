---
Reference: "[[https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/]]"
aliases:
  - "SAM 2: Segment Anything in Images and Videos"
---
## 목차
- [[#Introduction|Introduction]]
- [[#Related Work|Related Work]]
	- [[#Related Work#Image segmentation|Image segmentation]]
	- [[#Related Work#Interactive Video Object Segmentation (iVOS)|Interactive Video Object Segmentation (iVOS)]]
	- [[#Related Work#Video Object Segmentation (VOS)|Video Object Segmentation (VOS)]]
	- [[#Related Work#Video segmentation datasets|Video segmentation datasets]]
- [[#Task: promptable visual segmentation|Task: promptable visual segmentation]]
- [[#Model|Model]]
	- [[#Model#Image Encoder|Image Encoder]]
	- [[#Model#Memory attention|Memory attention]]
	- [[#Model#Prompt Encoder|Prompt Encoder]]
	- [[#Model#Mask Decoder|Mask Decoder]]
	- [[#Model#Memory Encoder|Memory Encoder]]
	- [[#Model#Memory Bank|Memory Bank]]
	- [[#Model#Training|Training]]
- [[#Data|Data]]
	- [[#Data#Data engine|Data engine]]
		- [[#Data engine#Phase 1: SAM per frame|Phase 1: SAM per frame]]
		- [[#Data engine#Phase 2: SAM + SAM 2 Mask|Phase 2: SAM + SAM 2 Mask]]
		- [[#Data engine#Phase 3: SAM 2|Phase 3: SAM 2]]
		- [[#Data engine#Quality verification|Quality verification]]
		- [[#Data engine#Auto masklet generation|Auto masklet generation]]
		- [[#Data engine#Analysis|Analysis]]
	- [[#Data#SA-V dataset|SA-V dataset]]
		- [[#SA-V dataset#Videos|Videos]]
		- [[#SA-V dataset#Masklets|Masklets]]
		- [[#SA-V dataset#SA-V training, validation and test splits|SA-V training, validation and test splits]]
		- [[#SA-V dataset#Internal dataset|Internal dataset]]
- [[#Zero-shot experiments|Zero-shot experiments]]
	- [[#Zero-shot experiments#Promptable video segmentation|Promptable video segmentation]]
	- [[#Zero-shot experiments#Semi-supervised video object segmentation|Semi-supervised video object segmentation]]
	- [[#Zero-shot experiments#Image segmentation|Image segmentation]]
- [[#Comparison to state-of-the-art in semi-supervised VOS|Comparison to state-of-the-art in semi-supervised VOS]]
- [[#Conclusion|Conclusion]]

## Introduction
![[Pasted image 20250714110820.png]]
- Segment Anything (SA) 모델은 promptable image segmentation을 구현하는데 성공했으나, 현실 세계에서는 비디오 또한 이미지만큼 널리 사용되고 중요함 -> 이미지와 비디오 둘다 적용 가능해야 함
- Video segmentation은 **시공간적 범위** (spatio-temporal)을 결정하는 task를 가짐. 이는 이미지 분할 이상의 고유한 과제.
	- 비디오는 움직임, 변형, 가려짐, 조명 변화 등에 의해 entity의 모습이 크게 달라질 수 있음.
	- 카메라 움직임, 흐림, low resolution 등으로 인해 이미지보다 품질이 낮은 경우가 대부분.
	- 수많은 프레임을 처리해야 함.
	- SAM은 이런 걸 처리하는 데 충분하지 않음.
- SAM 2는 이미지와 비디오를 통합적으로 처리하는 것을 목표로 함.
- Promptable Visual Segmentation (PVS): image segmentation을 video domain으로 일반화함.
	- 각 시공간 마스크에 대한 segment of interest를 정의 (masklet)
	- memory를 사용해 연속적 프레임 정보를 저장
- 학습 데이터를 생성하기 위한 date engine 도입
	- 특정 category에 제한되지 않고 어떠한 객체든 분할
	- 이를 사용해 Segmenta Anything Video (SA-V) 데이터셋을 구축.

## Related Work
### Image segmentation
- Segment Anything (Kirillov et al., 2023)에서 promptable image segmentation을 소개한 바 있음
	- 사용자 프롬프트를 받아, 이미지에서 valid한 segmentation mask를 출력하는 foundation model
- 최근 연구 동향은 SAM을 확장하거나 효율성을 개선하는 방식으로 쓰임
### Interactive Video Object Segmentation (iVOS)
- 주석, 클릭, bounding box 등의 사용자 입력을 통해 비디오에서의 객체 분할 (**masklets**)을 얻는 task
- 최근 연구는 SAM에 video tracker를 결합한 방식으로 많이 사용
	- 이러한 방식은 트래커가 작동하지 않거나, 비디오 프레임에서의 성능 저하, 모델의 예측 오류를 수정한 방법 부족 등으로 한계점이 명확함
### Video Object Segmentation (VOS)
- 비디오의 첫 프레임에 object가 제시되었을 때, 해당 object를 비디오 전체에서 정확히 추적하는 task
	- Semi-supervised VOS라고도 부름
- 초기에는 online-finetuning을 사용해 모델을 target에 적응(adapt) 하도록 만듬.
- 이후 offline-trained model, RNN, Transformer를 사용해 성능 및 효율성을 개선함.
### Video segmentation datasets
- VOS task를 위한 datasets
- 기존 데이터셋으로 DAVIS, Youtube-VOS 등이 있음.
- 이러한 데이터셋들은 규모가 작거나 특정 category에 집중되어 있어 SAM울 훈련시키기에는 부적합하였음.
## Task: promptable visual segmentation
- Promptable Visual Segmentation (PVS): 기존 이미지에서의 Segment Anything을 video domain으로 확장한 것.
	- 비디오 전체에서 object를 segment하고 추적하는 것을 목표로 함.
- SAM 2에서 집중적으로 다루는 PVS task는, 비디오의 어떤 프레임에서든 prompt를 제공 가능.
	- 프롬프트는 clicks, boxes, masks 등이 될 수 있음
	- 프롬프트는 positive/negative일 수 있음. positive는 object 지정, negative는 모델 예측 수정
- Interactiva하게 동작.
	- 특정 프레임에서 프롬프트가 주어지면, 즉각적으로 해당 프레임에 대한 segmentation 결과가 제공되어야 함. 
	- initial prompt를 받으면, 이를 전파하여, 비디오 전체에 걸친 시공간적 mask를 예측해야 함. 전체 비디오에 대해, 각 비디오 프레임마다 segmentation mask를 예측해야 하는 것임. 이를 **masklet**이라고 칭함.
![[Pasted image 20250714113506.png]]
- 이미지에서 보면, frame 1에서 prompt를 통해 target object로 혀를 지정함.
- 첫 프레임에서 prompt를 받았으므로, 이를 비디오의 모든 프레임으로 전파함. step 1의 파란색 화살표를 따라, 첫 프레임에서 지정한 prompt가 모든 프레임으로 전파되고, 이 정보를 사용해 각 프레임마다 혀의 segmentation mask를 만들어냄. 
- 모델이 중간에 object를 놓쳤다면, 중간에 이를 수정할 수 있음. 그림의 frame 2 이후로 혀를 놓쳤을 때, 추가적으로 프롬프트를 주어 이를 수정하는 것이 가능. 혀를 놓친 프레임인 frame 3에서 추가적인 프롬프트를 주어 (빨간 화살표) mask를 수정하는 것이 가능. 이렇게 수정된 프롬프트 역시 다음 프레임들로 계속 전파되어 올바른 masklet이 만들어짐.
## Model
- SAM 2는 SAM의 개념을 비디오로 확장하여 객체를 시공간적으로 분할하는 것을 목표로 함.
- decoder에서 사용되는 frame embedding은 iamge encoder에서 바로 오지 않고, 과거 prediction과 prompt가 주어진 frame의 memory로부터 conditioned됨.
	- 현재 프레임 기준으로, prompted frame이 미래 시점일수도 있음.
	- Frame memory는 memory encoder에 의해 현재 예측을 기반으로 생성되어 memory bank에 저장됨.
	- memory attention: image encoder의 frame embedding과 memory bank간의 attention 연산. 해당 attention 결과를 기반으로 mask decoder가 예측을 생성.
![[Pasted image 20250714115213.png]]
### Image Encoder
- 비디오 프레임이 들어올 때 실행되어 feature embedding (=unconditioned tokens)을 추출.
	- 비디오 프레임이 들어올 때 처음 한번만 실행됨.
	- streaming approach를 사용, 비디오 프레임이 모델에서 처리 가능할 때 순차적으로 즉시 처리. 이를 통해 긴 비디오도 실시간으로 처리 가능.
	- MAE로 pretrained된 Hiera image encoder를 사용. 계층적이기 때문에, decoding 시 multiscale feature를 사용할 수 있음.
- stride 16, 32 feature(high-level feature)는 memory attention에 사용. stride 4, 8 feature(low-level feature)는 mask decoder의 upsampling 과정에서, 고해상도 detail을 위해 추가됨 (아래 Figure 8 참고)
### Memory attention
- 현재 프레임의 feature를 과거 프레임의 fetaure 및 prediction과 결합하는 역할.
- L개의 transformer block을 쌓아 구현
	- 첫 번째 블록은 현재 프레임의 image encoding을 input으로 받음.
	- 각 블록들은 self-attention, (prompt 유무와 관계없는) frame memory와 objet pointer에 대한 cross-attention, 그리고 MLP를 수행함.
### Prompt Encoder
- SAM과 동일한 구조.
- clicks, boxes, masks 등의 prompt를 통해 해당 프레임에서의 object 범위를 지정 가능함.
### Mask Decoder
![[Pasted image 20250714150750.png]]
Figure 8: Mask decoder 구조
- "two-way" transformer 블록을 쌓아, 프롬프트와 frame embedding을 업데이트 가능하도록 함.
- SAM과 비슷하게, 모호한 프롬프트(ex. single click)에 대해 여러 개의 마스크를 예측 가능함.
	- 모델이 valid한 mask를 출력하기 위해 중요한 디자인임.
	- video에서는, 모호성이 각 비디오 프레임마다 전파될 수 있으므로, 각 프레임에 대해 여러 mask를 예측.
- SAM과 다른 점은 주어진 positive prompt를 만족하는 object가 항상 있다고 보장할 수 없음. (object가 가려지거나 하는 경우가 있을 수 있음)
	- 이를 처리하기 위해 새로운 head 도입: object가 현재 프레임에 존재하는지 예측
- decoding 과정에서 high-resolution embedding을 얻기 위해 image encoder로부터의 skip connection 또한 활용.
### Memory Encoder
- 출력 마스크와 이미지 인코더의 feature를 결합하여 메모리를 생성.
	- mask decoder의 output mask를 downsampling하고, unconditioned frame embedding과 element-wise sum을 통해 memory를 생성.
	- 추가적인 encoder를 사용하지 않고, image encoder의 embedding을 사용
### Memory Bank
- 최근 N개 frame memory와 M개의 prompted frame 정보를 저장.
	- FIFO queue 형태로 저장.
	- VOS task를 예시로 들면, 첫 프레임에서만 prompt가 주어지므로, 첫 프레임의 memory, 그리고 최근 N개의 frame memory를 저장하게 됨.
	- 공간적 feature map 형태로 저장.
- spatial memory뿐만 아니라 object pointer도 함께 저장.
	- object의 high-level semantic information을 담은 vector
	- 각 프레임에서의 mask decoder output token을 기반으로 생성.
	- memory attention은 공간 정보와 object pointer 간의 cross-attention을 수행
- object의 움직임을 추적하기 위해 최근 N개 frame의 temporal position information도 함께 저장. 
### Training
- image와 video를 joint하게 학습.
- interactive prompting을 시뮬레이션함.
	- 8프레임 sequence를 추출하여 랜덤하게 2개 프레임을 prompting함
	- ground-truth masklet과 모델 예측을 사용하여 corrective click을 샘플링하고, 이를 확률족으로 적용.
- 학습은 ground-truth masklet을 sequential하게, 그리고 interactive하게 예측하는 것이 목표.
- 초기 prompt는 gt mask, gt mask로부터 샘플링된 positive click, 그리고 bounding box 중 랜덤하게 선택됨
## Data
- related work에 나와 있던 것처럼 기존 video segmentation 데이터셋들은 Segment Anything의 철학과는 맞지 않음 -> 대규모 데이터셋 수집을 위한 data engine을 구축
- **interactive model in the loop setup with human annotators**: 인간과 모델이 상호작용하는 관계
- 전체 object와 그 부분까지 잡아내는 것을 목표로 함.
### Data engine
- data engine은 세 개의 phase로 구성.
#### Phase 1: SAM per frame
- image-based interactive SAM을 사용
- annotator가 SAM을 사용하여 target object를 직접 annotate함 (6 FPS)
- 모든 프레임을 처음부터 직접 labeling해야 했음. 이 때문에 프레임당 평균 37.8초가 소요될 정도로 느렸음.
- 그러나 고품질의 spatial annotation을 얻을 수 있었음.
- 약 1,400개 video에서 16,000개 가량의 masklet을 수집.
#### Phase 2: SAM + SAM 2 Mask
- Phase 1의 loop에 SAM 2를 도입, SAM 2에는 오직 mask만 프롬프트로 줄 수 있도록 함 (~SAM 2 Mask)
- annotator가 SAM을 사용해 비디오 첫 프레임의 spatial mask를 생성하면, SAM 2 Mask를 사용해 temporal하게 다른 프레임으로 전파 -> spatial-temporal masklet을 얻을 수 있음
- 잘못된 부분이 있으면 어느 프레임에서든 annotator가 spatialy하게 prediction을 수정할 수 있음.
	- 이 때 SAM을 사용하여 해당 프레임에서 mask를 다시 생성하여 다시 propagate
- SAM 2 Mask는 Phase 1 데이터와 public dataset으로 학습됨.
- 약 63,500개 가량의 masklet을 수집. time은 프레임당 약 7.4초로 5배 가량 빨라짐.
- 빨라지긴 했으나 중간에 처음부터 다시 annotate 해야한다는 문제는 여전함.
#### Phase 3: SAM 2
- fully-featured SAM 2를 활용 (mask 외에도 다양한 프롬프트 활용)
- 시간 차원에서의 object memory를 사용하므로 mask를 예측함에 있어 이점이 있음
- annotator는 수정을 위해 간단한 클릭만 수행하면 됨 (이전 Phase 2에서는 수정하려면, 수정하려는 프레임에서부터 다시 SAM을 사용해 mask를 잡아주어야 했음)
	- 이것은 memory가 있기 때문에 가능
- 이렇게 수집된 annotation을 사용해 다섯 번 반복 학습
- 약 197,000개 masklet을 수집, 프레임당 4.5초 가량 소요되어 Phase 1에 비해 약 8배 속도 향상됨.
#### Quality verification
- annotator들이 각 masklet의 quality를 만족/불만족 으로 분류
	- 만족(satisfactory): 모든 프레임에 걸쳐 target object가 잘 추적됨
	- 불만족(unsatisfactory): terget object는 잘 잡혔으나 masklet이 올바르지 않음
	- object가 잘 잡히지 않은 masklet은 완전히 reject.
- 불만족 masklet은 refinement pipeline을 통해 수정됨.
#### Auto masklet generation
- human annotator는 인간 편향이 있을 수 있음 -> 자동으로 masklet을 생성하여 augment를 수행
	- annotation의 커버리지를 넓히고 failure case를 명확히 하는 데 도움이 됨
- SAM 2에, 첫 번째 프레임에 격자점을 프롬프트로 주어 후보 masklet을 생성하도록 함
	- 이렇게 생성된 masklet은 verification 과정을 거침
	- "만족"으로 분류된 masklet은 데이터셋에 추가됨.
	- "불만족"으로 분류된 masklet은 annotator에게 전달되어 Phase 3를 거치게 함.
- 명확히 보이는 가운데 object뿐 아니라 뒤에 있는, 다양한 사이즈의 object들도 잘 잡히도록 함.
#### Analysis
![[Pasted image 20250714135604.png]]
Table 1: Data engine의 각 Phase에 대한 비교. 
- Phase 1 Mask Alignment Score: Phase 1의 결과와 비교하여 IoU > 0.75 이상인 샘플의 비율
- Phase 1은 사람이 직접 처음부터 끝까지 labeling한 것이므로 reference로 삼음.
- Phase 3의 결과가 모든 면에서 제일 좋았음.

![[Pasted image 20250714135612.png]]
Table 2: 각 데이터셋을 사용해 SAM 2를 학습시켰을 때의 성능 비교
- J&F accuracy metric (higher is better)
- 각 phase의 데이터셋을 추가할 때마다 모델 성능이 좋아지는 경향
	- in-domain 데이터셋인 SA-V뿐만 아니라 zero-shot tasks에서도 성능이 올랐다는 것이 주목할만한 점
### SA-V dataset
- Data engine을 사용해 만들어진 VOS 데이터셋
	- 약 50,900개 비디오에서 642,600개 가량의 masklet을 포함
![[Pasted image 20250714140444.png]]
- 기존 데이터셋들보다 53배 이상 큰 사이즈
#### Videos
- 약 50,900개 video
- 내부 영상 54%, 외부 영상 46%, 평균 길이 약 14초
- 다양한 환경, 일상생활을 담고 있음
#### Masklets
- annotator가 생성한 190,900개 가량의 masklet
- data engine에 의해 자동으로 생성된 451,700개 가량의 masklet
- disappearance rate: oebject가 사라지는 비율이 높아 난이도가 높은 dataset
![[Pasted image 20250714140832.png]]
![[Pasted image 20250714140845.png]]
Fig 4: SA-V dataset의 masklet 예시
#### SA-V training, validation and test splits
- 유사한 객체의 중복을 최소화하기 위해 video author를 기준으로 split
- validation 및 test set을 만들기 위해 어려운 scenario에 집중
	- fast-moving, complex occlusion 등의 어려운 타겟을 annotator가 직접 선별
	- 이 데이터들은 Phase 1에서 6 FPS로 annotated됨
- SA-V val split: 155 video, 293 masklets
- SA-V test split: 150 video, 278 masklets
#### Internal dataset
- 내부적으로만 접근 가능한 licensed video를 사용
## Zero-shot experiments
- 기존 모델과 SAM 2의 성능을, zero-shot video 및 image task로 비교
- 비디오는 J&F metric, 이미지는 mIoU
### Promptable video segmentation
- PVS task의 목적: user-guided prompt를 사용해 objet of interest를 지정하고 masklet을 일관되게 예측하는 것
- interactive setting을 시뮬레이션하는 방식으로 평가
- 두 가지 settings 사용
	- *Offline* Evaluation: 비디오를 여러 번 탐색하며 error가 가장 큰 프레임을 선택하여 수정
	- *Online* Evaluation: 비디오를 한 번만 보고 low-quality 프레임이 있으면 일시중지하고 해당 프레임을 수정
- 9개의 densely annotated zero-shot video에 대해 수행 (N_click: 프레임 당 3번의 클릭 프롬프트)
![[Pasted image 20250714142528.png]]
- 프롬프트를 준 프레임 수(N_frame)에 따른 J&F metric을 비교
- 기존 SOTA 모델에 비해 SAM 2가 약 3배 적은 interaction으로 더 좋은 성능을 보여줌
### Semi-supervised video object segmentation
- 첫 번째 프레임에만 prompt를 주어 VOS를 수행
	- 클릭을 사용할 때, 첫 프레임에 1개, 3개, 5개의 클릭만 사용
![[Pasted image 20250714143117.png]]
Table 4: 17개의 video dataset에 대한 segmentation 성능 비교
- 모든 경우에서 SAM 2가 더 나은 성능
	- SAM 2가 non-interactive한 task에서도 잘 동작함을 보여줌
	- 다른 모델들은 VOS에 특화되도록 구성된 모델임이 주목할만한 점
### Image segmentation
- 37개의 zero-shot dataset으로 구성된 Segment Anything task에 대한 평가
	- 이 중 23개는 SAM의 평가에 사용됨
![[Pasted image 20250714143435.png]]
Table 5: SA task에 대한 zero-shot accuracy
- SA-23 All: SAM에 사용된 23개 데이터셋
- Encoder의 개선으로 인해 동일 조건에서 SAM 2가 SAM보다 조금 더 나은 성능을 보여줌.
- Data engine을 사용하여 만든 데이터셋을 사용했을 때 성능이 크게 오르는 현상을 보여줌.
## Comparison to state-of-the-art in semi-supervised VOS
- SAM 2의 주요 focus는 PVS task이지만, 역사적으로 흔한 protocol인 semi-supervised VOS setting과도 비교해봄
- encoder size가 다른 두 개의 SAM 2 (Hiera-B+/Hiera-L)을 사용해 speed-accuracy tradeoff를 평가
![[Pasted image 20250714144133.png]]
Table 6: 기존 모델과 SAM 2의 VOS 성능 비교
- 기존 모델에 비해 높은 성능
- image encoder가 클 수록 성능 향상폭이 컸음
- SA-V 데이터셋에 대해 SAM 2가 매우 높은 정확도를 달성
	- 기존 방법들이 "segment anything" 능력이 취약함을 보여줌
## Conclusion
- 기존 Segment Anything task를 세 가지 방법을 통해 video domain으로 성공적으로 옮겨옴.
1. promptable segmentation task를 video로 확장
2. SAM 아키텍쳐에 메모리를 적용
3. 학습과 벤치마깅에 SA-V 데이터셋을 적용
