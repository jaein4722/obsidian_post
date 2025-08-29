---
aliases:
  - |-
    Q-SAM2: Accurate Quantization for Segment
    Anything Model 2
Reference: "[[https://www.arxiv.org/pdf/2506.09782]]"
---
## 목차

- [[#Introduction|Introduction]]
- [[#Background and Related Work|Background and Related Work]]
	- [[#Background and Related Work#Segment Anything 2 (SAM2)|Segment Anything 2 (SAM2)]]
	- [[#Background and Related Work#Model Quantization|Model Quantization]]
	- [[#Background and Related Work#Quantization Aware Training (QAT)|Quantization Aware Training (QAT)]]
- [[#Method|Method]]
	- [[#Method#Challenges in quantizing the SAM2 Image Encoder|Challenges in quantizing the SAM2 Image Encoder]]
	- [[#Method#Linear Layers Calibration for Initializing Quantized SAM2|Linear Layers Calibration for Initializing Quantized SAM2]]
	- [[#Method#Quantization-Aware Training for Finetuning Quantized SAM2|Quantization-Aware Training for Finetuning Quantized SAM2]]
- [[#Experiments|Experiments]]
	- [[#Experiments#Experimental setup|Experimental setup]]
		- [[#Experimental setup#Task, Dataset, and Metrics|Task, Dataset, and Metrics]]
		- [[#Experimental setup#Implementation|Implementation]]
	- [[#Experiments#Ultra-Low-Bit Quantization Performance|Ultra-Low-Bit Quantization Performance]]
	- [[#Experiments#Calibration in Post Training Quantization|Calibration in Post Training Quantization]]
- [[#Conclusion|Conclusion]]

## Introduction
- SAM 2는 promptable visual segmentation이라는 어려운 task를 성공적으로 이루어냄 (상세 참조: [[SAM 2 Segment Anything in Images and Videos]])
	- 그러나 SAM 2는 계산 및 메모리 소비량이 매우 많으므로 실시간 추론을 위해 A100 같은 고성능 하드웨어가 필요함 -> 엣지 디바이스와 같은 환경에서 돌리기에 제약이 많음
- 이러한 큰 모델을 작게 만드는 효과적인 방법 중 하나는, 적은 비트로 양자화(quantize)하는 것. 
	- 대표적인 방법으로 PTQ, QAT가 있음.
		- Post-Training Quantization (PTQ): 최근 LLM을 경량화하는데 많이 쓰임. 재학습 없이 계산 비용을 줄이고 가속화 가능.
		- Quantization-Aware Training (QAT): training 과정에서 quantization의 영향을 시뮬레이션하여, 적은 precision bit로도 정확도를 복구할 수 있게 해줌.
	- 그러나 SAM 2에는 이러한 경량화 기법이 많이 연구되지 않음
		- 기존 SAM에 대한 PTQ 알고리즘이 제시된 적은 있으나, 더 까다로운 task를 수행하는 SAM 2 특성상 PTQ만으로는 최적화가 어려울 수 있음.
		- SAM 2의 quantization은 도전적인 task. weight와 activation에서 이상치가 다수 있음.
- 따라서 본 논문에서는 SAM 2를 효과적으로 양자화하기 위한 파이프라인을 제안. 특히 2bit 수준까지의 초저비트 양자화를 목적으로 함.
___
![[Pasted image 20250715130421.png]]
Figure 1: Q-SAM 2의 접근법 도식. calibration batch를 사용해 weight 분포를 보정. 이렇게 보정된 image encoder를 대치하는 방식으로 접근.

## Background and Related Work
### Segment Anything 2 (SAM2)
- SAM 2는 Meta AI(구 페이스북)에서 개발한, 이미지와 비디오를 모두 처리 가능한 promptable segmentation 모델
- SAM 2의 초점은 video -> hierarchical encoder와 streaming memory mechanism을 활용해 프레임 간의 시공간적 정보를 보존, re-prompting 없이 효과적으로 mask를 추적 가능.
- Q-SAM2에서의 주 focus는 SAM 2의 image encoder. 전체 파라미터의 90% 이상을 차지.
### Model Quantization
- Quantization: 연속적인 고해상도의 discrete domain을 잘 표현하는 finite set을 근사하여 찾는 과정.
- 성능 저하를 최소화하면서 computational efficiency를 챙기는 것이 목표
$$\mathbf{X}_q = \text{clip}\left(\text{round}\left(\frac{\mathbf{X}}{s}\right) + z, 0, 2^b - 1\right).$$
- $\mathbf{X}$: FP16, FP32 등 높은 정밀도로 표현된 input matrix, 또는 tensor
- $z$: zero point. 실수 0이 mapping되는 값.
- $b$: quantization 비트 수. 해당 값에 따라 quantization 범위가 달라짐.
	- unsigned: $[0, 2^b - 1]$
	- signed: $[-2^{b-1}, 2^{b-1} - 1]$
	- 범위를 넘지 않도록 적절히 clipping해주어야 함.
- $s$: scaling factor. 값을 quantization 범위로 mapping하는 역할.
- $z, s$는 양자화 목적에 따라 달라짐
	- tensor quantization일 경우 scalar, per-axis quantization일 경우 vector, group quantization일 경우 tensor
### Quantization Aware Training (QAT)
- 학습 과정에서 fake quantization operation을 사용해 quantization의 영향을 시뮬레이션함
- $z, s$를 learnable parameter로 두고 학습 과정에서 동적으로 조정
- quantization은 finite한 set으로 mapping하는 과정이기에, 미분 불가능
	- 따라서 Straight-Through Estimator (STE)를 통해 gradient를 업데이트
---
(참고): Straight-Through Estimator: discretization 과정을 무시하고 원 출력 값을 전달하여 update하는 방법
![[Pasted image 20250715132449.png]]
___
- PTQ는 original weight에 추가적인 modify를 거치지 않고 quantization을 적용함
	- 이 때 $z, s$는 calibration data를 사용해 reconstruction error가 가장 작아지는 값을 계산해 사용
	- 간단히 적용할 수 있지만 performance degradation 폭이 큼
- 따라서 학습 과정에서 양자화를 시뮬레이션하는 QAT가 더 높은 정확도를 달성할 수 있음
	- 특히 low-bit일수록 이러한 현상이 더 두드러짐
	- 그러나 training complexity가 증가한다는 단점이 있음
- QAT를 low precision에서 적용하며 높은 정확도를 이끌어낼 수 있는 방법이 다수 제시됨
	- DoReFa-Net: 초기에 많이 사용된, 결정론적 함수를 통해 양자화를 도입하는 방법.
	- PACT: learnable한 clipping parameter를 도입하여 range를 제어하려는 시도
	- LSQ: backpropagation을 통해 quantization step을 학습

## Method
### Challenges in quantizing the SAM2 Image Encoder
- SAM2에서 사용된 Hiera-based image encoder는 기존 transformer 아키텍처보다도 quantization을 적용하기 어려움.
	- weight의 분포가, 매우 큰 소수의 outlier로 인해 heavy-tailed 형태를 띠고 있음. 또한 이러한 값들이 불규칙하게 등장.
	- 일반적인 quantization 방법으로는 이러한 극단값을 표현하기 어려움.
	- 특히 더 큰 outlier들이 초기 레이어에 집중되어, 전파될수록 에러가 퍼짐.
	- weight 비트 폭을 4비트 이하로 줄이면 제대로 segmentation이 되지 않을 정도였음.
- 해결을 위한 실마리로 이러한 극단 값을 clipping으로 잘라내는 것이 도움이 되었음.
	- 처음에는 정확도가 많이 손실되었지만, 모델의 핵심적인 prediction 행동은 보존됨 -> outlier가 모델에 큰 영향을 미치지 않는다는 것을 시사
	- 따라서 이러한 꼬리 값(tail value)를 잘라내는 것이 quantization에 도움이 되었음.
---
![[Pasted image 20250715135603.png]]
Figure 2. 3-bit quantization에 대해 Clipping을 수행했을 때의 distribution 영향.
- (왼쪽) clipping을 하지 않았을 때는 가중치 분포가 훨씬 느리게 감소하는 heavy-tailed 형태. 이로 인해 분포가 매우 넓게 퍼져 있으며, quantization 분포 또한 특정 영역에만 집중되어 있음. 이는 표현력 감소의 주된 원인.
	- 이렇게 발생한 초기 quantization error는 finetuning을 하더라도 쉽게 복구되지 않음.
- (오른쪽) clipping을 적용한 이후에는 분포가 비교적 고르게 퍼져 있으며, quantization level 분포도 매우 고른 것을 볼 수 있음.
### Linear Layers Calibration for Initializing Quantized SAM2
- 핵심 아이디어는 QAT에서도 calibration을 위한 **preprocessing**을 수행하는 것
	- calibration 단계는 PTQ에서는 중요하지만, QAT에서는 지금까지는 중요히 다뤄지지는 않았음
	- SAM2 모델의 linear layer의 weight variance를 평균값 근처로 조정하여 초기 quantization error를 줄이는 것.
	- 작은 calibration set을 사용해, Frobenius norm을 최소화하는 가중치 matrix를 찾는 방식
	- weight의 평균이 작을 때, 행렬의 Frobenius norm이 상수 factor로 분산을 근사할 수 있어 전체 weight 크기를 측정하기 적합함.
	- 이미 학습된 weight matrix를 건드린다는 것 자체로 performance degradation 위험이 있음. 따라서 matrix를 변경하더라도 linear layer의 출력 자체는 최대한 유지되어야 함.
- 따라서 이러한 조건을 충족하기 위해, calibration batch로부터 span된 subspace에 weight를 projection하는 형태로 초기 calibration step을 수행
	- **쉽게 말하면, input activation $X$와 output activation $Y$에 대해, $Y=XW^T$를 만족하는 새로운 가중치 행렬 $\hat{W}$을 찾는 것임. 이때 $\hat{W}$의 Frobenius norm이 최소가 되는 해를 찾는게 궁극적인 목적.**
	- 원 모델에 calibration batch를 통과시켜, input activation과 linear layer output 간의 Moore-Penrose Pseudoinverse를 학습하여 계산하는 방식으로 구현
		- 따라서, 찾고자 하는 $\hat{W}$는 $X$와 $Y$의 Moore-Penrose Pseudoinverse를 구하면 얻을 수 있음.
		- 원본과 양자화 결과 간 잔차를 최소화할 수 있으며, 이렇게 구한 weight의 Frobenius norm이 최소가 되기 때문에 Moore-Penrose Pseudoinverse를 사용.
- 그렇다면 calibration batch에 대해서만 weight를 구하면, 나머지 샘플에서는 성능이 감소할 수 있는 것 아닌가?
	- calibration된 가중치 행렬의 특이값을 제어하여, 스펙트럼에서 불필요한 값을 제거할 수 있음. 쉽게 말해 동작을 더욱 간단하고 일반적으로 만들 수 있다는 뜻. 이러한 일반적인 동작은 quantization을 거치더라도 robust하게 동작할 것.
	- 그럼에도 불구하고 calibration batch를 잘못 선택하면 전체 데이터에 대한 generalization이 잘 이루어지지 않음. 따라서 다양하고 조건이 좋은 sample을 잘 선택해야 함.
	- 조건이 좋은 sample임을 나타내는 condition number를 도입: 값이 낮을수록 안정적이고 풍부한 subspace를 담고 있음을 나타냄.
- 수식적 정의
	- input tensor $\mathbf{X}\in\mathbb{R}^{n\times d_{in}}$, output tensor $\mathbf{Y}\in\mathbb{R}^{n\times d_{out}}$, batch size $n$
	- linear layer $f(\mathbf{X}, \mathbf{W}) = \mathbf{XW^\top} + \mathbf{b}$, $\mathbf{W}\in \mathbb{R}^{d_{out}\times d_{in}}, \mathbf{b}\in \mathbb{R}^{d_{out}}$
	- estimated weight matrix $\hat{\mathbf{W}}^\top = \mathbf{X}^\dagger\mathbf{Y}$ ($\mathbf{X}^\dagger = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}$는 $\mathbf{X}$의 Moore-Penrose Pseudoinverse, bias는 생략)  
	- Tikhonov regularization 추가: $\hat{\mathbf{W}}^\top = (\mathbf{X}^\top\mathbf{X}+\lambda I)^{-1}\mathbf{X}^\top\mathbf{Y}$, $\lambda$는 cutoff hyperparameter, L2 regularization 역할을 하여 calibration batch에 대한 overfitting을 방지
---
![[Pasted image 20250715151222.png]]
Figure 3. calibration 이후 heavy-tailed 현상이 완화되고 variance가 줄어 dynamic range를 더욱 압축할 수 있음. 이는 quantization error를 줄이는 데 중요함
### Quantization-Aware Training for Finetuning Quantized SAM2
- linear layer의 calibration과 clipping 이후 QAT를 수행
- QAT 수식
$$\mathbf{X}_q = \text{clip}\left( \text{round}\left( \frac{\text{clip}(\mathbf{X}, \mu_{\mathbf{X}} - \alpha \cdot \sigma_{\mathbf{X}}, \mu_{\mathbf{X}} + \alpha \cdot \sigma_{\mathbf{X}})}{s} \right) + z, -2^{b-1}, 2^{b-1} - 1 \right)$$
- $\mu_{\mathbf{X}}, \sigma_{\mathbf{X}}\in\mathbb{R}^n$은 batch size에 따라 $\mathbf{X}$의 통계량, 혹은 input channel의 통계량이 될 수 있음
- 초기 clipping의 정도는 cutoff factor $\alpha$로 조절
![[Pasted image 20250715151638.png]]
Figure 4. SAM2 QAT의 전체 흐름.

## Experiments
### Experimental setup
#### Task, Dataset, and Metrics
- 실험 설정은 SAM2와 거의 동일
- SA-1B, SA-V 데이터셋을 사용해 calibration 및 training 수행
	- 전체 데이터셋을 모두 사용하지 않고 작은 subset만 사용. full retraining 없이 높은 정확도 달성 가능.
- Semi-supervised VOS(J&F metric)과 instance segmentation(mIoU)에서 evaluation 수행
#### Implementation
- SAM2 checkpoint로부터 파인튜닝 수행
	- sam2.1_hiera_tiny(T), small(S), base_plus(B+) 사용
- calibration: SA-1B로부터 50장의 이미지를 랜덤 샘플
	- 50장의 calibration batch에 대한 activation으로부터 calibrated weight matrices 계산
- 비교를 위한 Baseline: 고전적인 MinMax 방식
- PyTorch 2.6.0, CUDA 12.4, A100 80GB
### Ultra-Low-Bit Quantization Performance
![[Pasted image 20250715152805.png]]
Table 1. Evaluation 결과.
- Q-SAM2 방법이 base_plus와 small 모델에서 기존 MinMax 방식보다 좋은 성능을 보임
	- 특히 W2A4 세팅에서 성능 향상 폭이 컸음
- Tiny 모델에서는 성능 향상 폭이 미미함. 이는 Tiny 모델이 다른 큰 모델들보다 outlier가 적어 clipping으로 인한 영향이 적으며, 잘 conditioned된 activation으로 인해 calibration의 영향이 줄어든 탓으로 보임.
	- Limitation: 비트폭이 충분하거나 파라미터 수가 적은 경우에는 Q-SAM2가 기존 방식과 비슷하거나 오히려 성능이 떨어짐.
![[Pasted image 20250715154746.png]]
Figure 5. B+ 모델에 대한 W2A4에서의 Qualitative results
### Calibration in Post Training Quantization
- calibration의 영향을 평가하기 위해 기존 PTQ 방법론에 calibration method를 적용하여 비교
![[Pasted image 20250715154959.png]]
Table 2. PTQ에 Q-SAM calibration을 적용했을 때 성능 비교
- 두가지 PTQ 방법론에 대해 calibration 적용 시 성능이 향상됨. 특히 3bit 구성에서 성능 향상 폭이 높음.
## Conclusion
- Q-SAM2는 SAM2에 quantization을 적용한 첫 번째 모델
- calibration과 clipping을 통해 quantization error를 최소화
	- 특히 2bit의 매우 낮은 비트에서도 QAT를 효과적으로 동작시킴
- 제안된 calibration은 최소한의 데이터만으로도 효과적으로 generalization을 달성.