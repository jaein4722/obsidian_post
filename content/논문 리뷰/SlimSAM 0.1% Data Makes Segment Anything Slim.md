---
aliases:
  - "SlimSAM: 0.1% Data Makes Segment Anything Slim"
---
## 목차

## Introduction
![[Pasted image 20250805123957.png]]
- SAM은 성능이 매우 좋으나 모델 사이즈와 높은 computational cost로 인해 리소스 제약 환경에서 사용하기 쉽지 않았음.
- 이전까지의 경량화 시도는 무거운 image encoder를 가볍게 교체, 구조를 효율적으로 변경하는 것 등이었음.
	- 그러나 이러한 방식은 모델은 처음부터(from scratch) 다시 학습시키기 때문에 학습에 많은 비용이 소요될 수 밖에 없음.
	- 이러한 이슈로 인해 새롭게 훈련된 경량 모델들은 original pre-trained SAM의 지식을 온전히 활용할 수 없음. 
	- pre-trained 지식을 최대한 활용하기 위해 pruning 기법을 적용한 사례도 있었지만, 큰 효과를 보지는 못함.
- SlimSAM은 이러한 SAM을 효율적으로 압축하기 위해 등장한 기법. 
	- 주요 기여는 alternate slimming framework, 그리고 disturbed Taylor pruning.
## Related Works
- **Model Pruning**
	- 모델의 파라미터를 직접적으로 줄여 (불필요한 파라미터를 없애) 경량화 및 속도 향상을 이루는 기법
	- Structural Pruning과 Unstructured Pruning으로 나뉨
		- structural: 파라미터 "그룹"을 특정 기준에 따라 제거
		- unstructured: 파라미터를 개별적으로 제거 (하드웨어 지원이 필수)
- **Knowledge Distillation**
	- 큰 모델의 지식을 작은 모델에게 넘겨주는 기법.
	- soft target function과 temperature parameter로 학습을 조절
- **SAM Compression**
	- SAM을 엣지 디바이스에 직접 올리기 위해 꾸준한 경량화 시도가 있었음.
	- FastSAM: SAM의 ViT 기반 아키텍쳐를 CNN 기반의 yolov8-seg로 교체
	- MobileSAM: 경량 Tinyvit로 encoder를 대체하고 원본 encoder로부터 KD를 수행
	- EdgeSAM: prompt-in-the-loop KD 개념을 이용, user input과 mask generation 간 관계를 잘 캐치함
	- EfficientSAM: MAE를 적용하여 효율적인 image encoder를 얻음. 그러나 학습에 필요한 데이터가 SA-1B보다도 많이 필요했다고 함.
	- 이러한 방법들 모두 scratch training을 수행하므로, 데이터가 적을 때 성능이 불만족스러움.
- **Remark**
	- 기존 SAM은 coupled structure를 가지고 있기 때문에 일반적인 pruning, KD 적용 시 성능이 크게 떨어질 수 있음.
	- 이에 alternate slimming network를 도입, 원본 모델과의 차이 최소화 & intermediate feature alignment를 가능하게 함
	- disturbed Taylor pruning을 사용해 pruning 목표와 training 목표 간 불일치를 해결함
## Method
- 본 연구의 목표는 large image encoder를 데이터가 적은 환경에서의 성능 저하를 최소화할 수 있도록 경량화하는 것이 목표.
- 이를 달성하기 위해 original SAM에서 핵심적인 weight를 추정: 이를 통해 1천 1백만개의 이미지로 학습된 SAM의 사전 지식을 충분히 활용할 수 있음
### Identifying SAM Redundancy
- 첫 번째 phase는 각 파라미터의 importance를 추정하는 것. 덜 중요하고 중복되는 파라미터를 제거하는 것이 목표.
- 파라미터를 제거했을 때의 prediction error를 정량화하여 계산
	- 즉, 해당 파라미터를 제거했을 때 loss 값이 기존과 크게 다르지 않아야 함
- $N$ image pair $\left\{x_i, y_i\right\}_{i=1}^{N}$, $\mathrm{M}$ parameters $W=\left\{ w_i \right\}_{i=1}^{N}$을 가진 모델 $\mathcal{F}$에 대한 $w_i$의 importance는 다음과 같이 정의 
$$
I_{w_i} = \left|\Delta \mathcal{L}(x_i, y_i)\right| = \left|\mathcal{L}_{w_i}(x_i, y_i) - \mathcal{L}_{w_i=0}(x_i, y_i)\right|
$$
- 이 때 파라미터 $w_i$ 부근의 loss 값 $\mathcal{L}_{w_i=0}$은 다음과 같은 first-order Taylor expansion을 사용해 근사 가능
$$
\mathcal{L}_{w_i=0}(x_i, y_i) = \mathcal{L}_{w_i}(x_i, y_i) - \frac{\partial \mathcal{L}(x_i, y_i)}{\partial w_i} w_i + \mathcal{R}_1(w_i = 0)
$$
- 식을 정리하여, 다음과 같이 importance를 근사 가능
$$
I_{w_i} \approx \left| \mathcal{L}_{w_i=0}(x_i, y_i) - \mathcal{L}_{w_i=0}(x_i, y_i) + \frac{\partial \mathcal{L}(x_i, y_i)}{\partial w_i} w_i \right| = \left| \frac{\partial \mathcal{L}(x_i, y_i)}{\partial w_i} w_i \right|
$$
- 그러나 이러한 접근이 SAM의 encoder를 pruning할 때는 두 가지 문제점이 있음.
	- taylor importance의 정확도가, hard label $y_i$에 크게 의존하는 형태
		- KD를 적용하기 위해선 hard label보다 soft label, 즉 image embedding에 기반한 값을 사용해야 함.
	- taylor importance를 구하는 과정에서 loss의 불일치 발생
		- importance 추정은 hard label에 대한 loss 최소화를 원하지만, distillation 기반 추정에서는 soft label에 대한 최소화를 원함
- 따라서 이러한 문제를 해결하기 위해 **disturbed Taylor importance**를 도입.
- 즉 hard label에 대한 loss를 구하는 것이 아닌, soft label에 대한 loss를 계산해서 사용
	- 그러나 input $x_i$와 embedding $t_i$간 gradient를 계산하면, 값의 차이에 의해 0이 나오게 될 것
	- 그래서 embedding에 gaussian noise를 추가. 즉, input과 "disturbed" embedding $t_i + \mathcal{N}(\mu, \sigma^2)$ 간 loss를 구하는 것. 이때 $\mu=0, \sigma=0.01$
	- 이게 가능한 이유는 disturbed embedding의 expectation $E(t_i + \mathcal{N})=t_i$이기 때문
$$
\begin{align*}
I_{w_i} &= |\Delta \mathcal{L}(x_i, t_i)| \approx |\Delta \mathcal{L}(x_i, t_i + \mathcal{N})| \\
&= |\mathcal{L}_{w_i}(x_i, t_i + \mathcal{N}) - \mathcal{L}_{w_i=0}(x_i, t_i + \mathcal{N})| \\
&\approx \left| \frac{\partial \mathcal{L}(x_i, t_i + \mathcal{N})}{\partial w_i} w_i \right|.
\end{align*}
$$
- **Remark**: disturbed Taylor importance를 통해 pruning의 objective와 distillation의 objective를 일치시킬 수 있었음. 77% pruning ratio에서 0.85%, 50% pruning ratio에서 0.60%의 mIoU 개선이 있었음. 또한 추가적인 연산 없이 전체 compression framework를 label-free하게 변경하였음.
### Alternate Slimming
- importance를 구했으면, 다음 목표는 이를 이용한 channel-wise structural pruning -> distillation-based finetuning을 수행.
- 일반적인 single-step pruning 기술로는, pruning 비율이 75%를 넘기면 성능이 기존에 비해 심각하게 저하되는 현상을 확인.
- 또한 데이터가 극히 적은 현재 환경 상, SA-1B 데이터셋의 0.1%만 사용하는 것으로는 distillation을 통해 성능을 크게 높이는 것에 한계가 있었음.
- 따라서 (1) 원본 모델과 pruned 모델의 성능 차이를 줄이면서 (2) distillation의 효과를 높일 수 있는 메커니즘이 필요 -> 이것이 alternate slimming framework.
![[Pasted image 20250805140902.png]]
- 크게 두 개의 sub-structure로 구성: embedding, bottleneck. 
	- embedding: 각 block의 output dimensions
	- bottleneck: 각 block의 intermediate features
- pruning과 distillation(각 sub-structure의 복원)을 순차적으로 진행하여 loss를 smooth하게 만듬
- 크게 4단계로 구성: embedding pruning, bottleneck aligning, bottleneck pruning, embedding aligning
#### Embedding Pruning
- embedding dimension은 encoder의 성능에 크게 영향을 줌 (encoder의 feature width를 결정)
- embedding dimension을 pruning하면서 bottleneck dimension은 일정하게 유지
	- 이때 residual connection이 있어 embedding dimension이 모든 블록에서 같아야 하므로, uniform local pruning을 사용
#### Bottleneck Aligning
- Embedding Pruning의 결과로 prune된 파라미터는 embedding space의 파라미터들 뿐임. 즉 bottleneck dimension은 기존과 동일
- embedding의 pruning으로부터의 손실을 복구하기 위해, 원본 인코더의 output으로부터 학습
	- dimensionality-consistent한 bottleneck feature인 원본 인코더의 intermediate feature에 맞게 align됨
	- 즉 pruned된 구조에서도 동일한 dimension을 가지는 bottleneck, 즉 intermediate feature의 값과의 차이를 줄이는 방향으로 파라미터를 조정하여 "align"시킨다고 해석 가능
$$
\mathcal{L}_{\text{Bn}} = \alpha \cdot \mathcal{L}_{\text{MSE}}(H_{v_0}, H_{v_1}) + (1 - \alpha) \cdot \mathcal{L}_{\text{MSE}}(t_{v_0}, t_{v_1})
$$
- 이 때 weight $\alpha$는 에포크에 따라 결정, 10에포크 전까지는 0.5, 그 이후는 0.
	- 즉 학습 초반에는 intermediate feature와 output을 모두 align하는 데 집중하고, 충분히 align되었을 거라고 판단한 학습 후반에는 output만 가지고 tuning하여 soft하게 학습되도록 함
#### Bottleneck Pruning
- 이전과 비슷하게, bottleneck dimension만 pruning하고 embedding dimension은 그대로 둠
- 이 때 bottleneck은 각 block들이 모두 독립되어 있으므로 각 block에 대해 다른 pruning ratio를 적용할 수 있음.
	- 처음 설정한 overall pruning ratio를 안넘는 선에서 자유롭게 조절이 가능하다는 뜻. 따라서 importance에 따른 global structural pruning 방식을 사용.
#### Embedding Aligning
- 이전과 비슷하게, bottleneck pruning 과정에서 차원이 유지된 embedding pruned 모델의 output에 맞게 align하게 됨.
	- 단, 기존과 달리 embedding pruned 모델로부터는 embedding, 그리고 최종 output에 모두 영향을 받음
	- 동시에, pruned되지 않은 original model과의 output에도 영향을 받아야 함
	- 따라서 최종 수식은
$$
\mathcal{L}_{\text{Emb}} = \alpha \cdot (\mathcal{L}_{\text{MSE}}(E_{v_1}, E_{v_2}) + \mathcal{L}_{\text{MSE}}(t_{v_1}, t_{v_2})) + (1 - \alpha) \cdot \mathcal{L}_{\text{MSE}}(t_{v_0}, t_{v_2})
$$
- 이 때 weight $\alpha$는 유동적으로 조절, $N=10$
$$
\alpha = \left\{
    \begin{array}{ll}
        \frac{N-n-1}{N} & n < N \\
        0 & n \ge N
    \end{array}
\right.
$$
- **Remark**: embedding-bottleneck으로 결합되어 있는 구조를 분리하여 독립적으로 pruning -> distillation 구조를 만들어냄. 특히 pruning ratio가 높을 때 성능 격차를 많이 줄일 수 있었음. ratio 각각 77%, 50%일 때 3.40%, 0.92%의 mIoU 향상이 있었음.
## Experiments

### Experimental Settings
- **Implementation Details**
	- single NVIDIA Titan RTX GPU로 훈련됨. SA-1B 데이터셋에서 0.1% (약 10,000개 이미지)만 사용하였음. base model은 SAM-B를 사용.
	- adam, batch 4
	- lr: initial 1e-4, 4 epoch 연속으로 진전 없을 시 절반씩 감소
	- SlimSAM-50 (50% pruning ratio)는 40epochs, SlimSAM-77 (77% pruning ratio)는 80epochs.
	- 그 외는 original 것을 그대로 사용.
- **Evaluation Details**
	- SA-1B 데이터셋의 GT와 prediction 간 mIoU를 비교
### Comparision and Analysis
#### Comparing with existing SAM compression methods
![[Pasted image 20250805145434.png]]
- 기존 SOTA 방법론들과 성능 비교
- original에 비해 4.0%, 1.4% 만의 파라미터, 3.5%, 0.8%만의 MACs만으로 SAM-H와 비슷한 성능을 기록
- 다양한 prompt에서도 디테일한 prediction을 도출 -> 기존 SAM의 robust한 사전 지식을 갖고 있음을 입증
- 데이터를 10,000개 (0.1%)만 사용했는데도 이정도 결과임이 주목할만한 점
#### Compraing with other structural pruning methods
![[Pasted image 20250805145844.png]]
- 각기 다른 pruning method를 적용했을 때의 성능도 비교 측정
- 전반적으로 성능이 훨씬 좋으며, 특히 pruning ratio가 높을 때 더 격차가 벌어짐
## Ablation Study and Analysis
#### Disturbed Taylor Pruning
![[Pasted image 20250805150330.png]]
- 기존 Taylor importance와 disturbed taylor importance 간의 성능 비교
- disturbed taylor pruning이 더 높은 정확도 달성
#### Intermediate Aligning
![[Pasted image 20250805150442.png]]
- distillation 과정에서 feature aligning의 효과 측정
- 중간 layer로부터 distillation을 해오는 것이 성능이 더 높았음
- 특히 step 1 distillation에서 그 성능이 많이 개선되었음
#### Alternate Slimming
![[Pasted image 20250805150652.png]]
- alternate slimming framework의 효과 분석
- 각기 다른 pruning 기준에 따라 alternate slimming 적용 전후로 성능을 비교했을 때 random, disturbed taylor 모두 성능이 높았음
#### Global Pruning vs Local Pruning
![[Pasted image 20250805151218.png]]
- bottleneck pruning에서 global pruning, local pruning의 성능 비교
- bottleneck dimension은 embedding과 달리 residual connection이 없어, 각 block이 모두 독립적으로 존재하고, 따라서 동일하게 dimension을 맞춰줄 필요가 없음
- local pruning은 성능이 항상 동일하였음. global pruning의 경우 group importance 설정에 따라 달랐지만, 특정 설정에서 upper-limit이 높아지는 경우 발견.
	- global pruning의 경우 이 importance를 normalize하여 선택하기 때문에 normalization 방법에 크게 의존하며, 해당 모델의 경우 실험적으로 gaussian normalization이 가장 성능이 좋았다고 함.
![[Pasted image 20250805151511.png]]
- 각 normalization 방법에 대한 pruning 이후의 intermediate dimension 시각화. 
- Min, Max, Gaussian의 경우 상대적으로 양 끝 값이 높고 중간값이 낮은 경향. 중간 layer들이 양 끝 layer보다 더 많은 redundancy를 보였기에 더 많이 제거되었다고 추측 가능. 즉 해당 normalization의 경우 중간 layer가 상대적으로 덜 중요할 가능성이 큼.
- Max, Standardization의 경우 일정한 패턴이 없음. 즉 redundancy를 제대로 캐치하지 못함을 나타냄.
#### Even less data
![[Pasted image 20250805152000.png]]
- 50% pruning에서는 데이터셋 감소가 성능에 큰 영향을 주지 않음
- 그러나 77%에서는 데이터셋 감소가 성능에 영향을 많이 줌
- 이는 더 높은 pruning ratio에서, 데이터셋을 늘리면 성능이 향상될 수 있다는 가능성을 시사함
#### Qualitative Results
![[Pasted image 20250805153140.png]]
![[Pasted image 20250805153148.png]]
## Conclusion
- SlimSAM은 최소한의 훈련 데이터로 높은 성능을 보여주는 SAM 압축 기법임.
- 핵심은 pre-trained SAM을 retraining 없이 재사용하는것. 
- 특히 복잡하게 연결된 embedding층과 bottleneck층을 따로 분리하여 pruning 및 distillation을 하는 alternate slimming framework가 원본 모델과의 격차를 줄이는데 많이 기여
## Limitations
- 학습 환경으로 인해 적은 수의 데이터셋으로만 학습됨. 10,000개만 썼는데, 학습 데이터를 더 많이 넣는다면 성능이 더욱 증가할 가능성이 있음. 저자들은 lossless compression이 가능할거라고 예측 중.
- SlimSAMS 방법론의 핵심은 원본 SAM 모델의 지식을 "재사용"한다는 데 있음. 따라서 원본 SAM 이상의 성능을 이끌어내는 것은 불가능.