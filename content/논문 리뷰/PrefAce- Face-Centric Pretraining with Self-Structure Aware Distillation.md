## 목차

- Introduction
- Related Work
- Proposed Method
- Experiment
- Conclusion

## Introduction

- 얼굴 분석은 인간의 비언어적 행동을 이해하는 데 중요한 통찰력을 제공.
- 사회적 상호작용, 의사소통, 인지 과정 등을 파악하는 데 큰 도움.
- 관련 분야: HCI(Human-Computer Interaction, 인간-컴퓨터 상호작용), Affective Computing(감성 컴퓨팅)
- 딥러닝 모델이 발전하면서, 얼굴 분석 관련 다양한 task들의 성능이 많이 올라옴. 대표적으로…
    - Facial Attribute Recognition (FAR): 얼굴 이미지나 영상에서 성별, 나이, 인종, 머리 색깔 등의 속성을 인식
    - Facial Expression Recognition (FER): 얼굴 표정에서 감정을 인식
    - DeepFake Detection (DFD): 딥페이크를 탐지
    - Lip Synchronization (LS): 음성 데이터와 입술의 움직임을 맞춤
- 이런 task들을 수행하기 위해서는, 대규모의 annotated dataset이 필요함 (특히 FER task는 전문 지식이 필요하다고 함.)

- Self-supervised pretraining: 레이블링되지 않은 데이터를 활용, 다양한 작업에 transfer가 가능한 generic representation 학습 가능, 이로 인해 레이블이 제한적이어도 학습을 가능하도록 함.
    - 실제로 natural scene imagery(자연 장면)에서는 이러한 self-supervised learning이 효과적이었음.
- 그러나 face analysis 분야에서는 대부분의 방법들이 특정 task에 특화 (transfer가 쉽게 불가능), fully-supervised learning에만 의존. 이로 인해 face analysis domain에서도 self-supervised pretraining 기법을 통합하고자 하는 시도가 여럿 있었음.
- 최근 연구에 의하면, 정제되지 않은 uncurated data에 self-supervised learning을 무턱대고 적용해버리면, transfer시키기 힘든 inferior representation(열등한 표현)을 얻게 된다고 함.
- SOTA를 찍은 최첨단 face 모델 (MARLIN)에서도 일반적인 common feature에만 집중하고, 더 discriminative한 feature는 오히려 간과해버리는 경향성을 보였음.
- 따라서, 해당 분야의 발전을 위해서는 uncurated data를 사용하면서도 common feature와 discriminative한 feature의 균형을 잘 맞추는 것이 포인트라고 할 수 있음.

- PrefAce는 self-supervised learning을 사용하면서도, 범용적이고 task에 구애받지 않는 representation의 학습에 치중을 두었다고 함.
- multiscale landmark guided self-distillaton: fine-grained discriminative한 facial representation의 학습을 목표로 함.
- FaceFeat cache: transferrable한 representation을 학습하기 위한 전략
- FAR, FER, DFD, LS 등 다양한 downstream task에서 sota를 능가했다고 함.
## Related Work
### Self-supervised Learning
- 레이블이 없는 대량의 데이터를 활용해, 모델 스스로 representation을 학습하도록 하는 방법
- 현실에서 대규모의 labeled dataset을 확보하기에는 시간과 자원이 많이 소요되므로, 데이터 내에서 모델이 스스로 signal을 생성하여 학습하도록 함.
- 일반적인 self-supervised learning 접근법
    - 데이터의 augmentations 간 일관성을 학습
    - dual encoder의 출력 distribution을 일치시켜 학습
    - input data의 일부분을 마스킹하고, 그 부분을 예측하도록 학습
### Facial Representation Learning
- 얼굴 이미지나 비디오에서 유용한 feature를 추출하는 방법을 학습
- 특정 task에 한정되지 않고, 좀 더 다양한 task에 일반적으로 적용할 수 있는 feature를 뽑는 것이 중요.
- 기존 방식들은 이러한 feature를 뽑기 위해, 특정 task만을 제한하여 fully-supervised learning을 도입하였음. 성능은 좋았지만, supervised learning인만큼 대규모의 annotated dataset이 필요.
- 이를 극복하기 위해 GAN 등을 사용해 synthetic data를 생성하는 방식의 연구가 활발하였음.
    - quantitative 문제는 해결했으나, qualitative 문제가 많았음.
- 최근에는 이를 해결하고자 하는 연구가 거의 없었음.
    - training dataset의 property를 개선하거나, visual-linguistic한 방식으로 해결하고자 함.
## Proposed Method
### Architecture
![[image.png]]
- **Dual Encoder**: teacher와 student의 두 개 네트워크로 구성.
    - 최근 self-supervised learning 연구에서 흔히 사용되는 구조로, 동일 데이터의 augmented view로부터의 정보 학습에 도움이 된다고 함.
    - 오직 video-level learning에만 집중함.
- **Multi-scale Processing**: 훈련 비디오 데이터셋 $\mathbb{D}=\left\{ V_i \right\}_{i=1}^{N}$에 대해, 각 비디오 $V\in \mathbb{R}^{C\times T \times H_0 \times W_0}$가 여러 스케일로 처리
    - 전체 비디오 클립이 입력으로 사용, global embedding $X_{global}\in \mathbb{R}^{\tfrac{T}{t} \times \tfrac{H_0}{h} \times \tfrac{W_0}{w}}$을 생성
    - global frame에서, 사전 정의된 얼굴 영역 v로부터 랜드마크 추적 → spatio-temporal(공간-시간적) 변화를 encoding / random하게 local tubes(연속된 프레임으로 구성됨)를 잘라냄
    - local frame(=tube) 또한 embedding $X_{local} \in \mathbb{R}^{\tfrac{T}{t} \times \tfrac{H}{h} \times \tfrac{W}{w}}$
    - local tube를 통해 landmark의 움직임에 따른 ‘skin’의 변화를 추적
    - 또한 이 영역에서, 주요 부분을 parsing (P={left eye, right eye, nose, mouth, hair})하여 따로 crop 후 저장 → skin detail에따른 landmark의 움직임과 encoding에 focus
    - landmark도 embedding, 이것이 최종 embedding이 됨 $X_{landmark} \in \mathbb{R}^{\tfrac{T}{t} \times \tfrac{H}{h} \times \tfrac{W}{w}}$
- **FaceFeat Cache**
    - downstream task를 위한 facial representation의 품질을 향상시키기 위한 기법, 일종의memory dict
    - 모든 video face instances의 representation을 저장
    - 학습 초기에, teacher network로부터, augmentation되지 않은 representation을 사용하여 메모리를 초기화
        - 각 video instance에 대해, 안정적이고 신뢰할 수 있는 baseline을 마련하기 위함.
    - 학습 중에는 momentum을 사용해 업데이트.
    - 동일한 instance의 다른 버전 representation 간 유사성을 maximize, 다른 instance 비디오와의 유사성을 minimize하도록 유도함 → 얼굴의 ‘특성’에 집중할 수 있도록 함.
        - 동일 instance의 다른 버전 representation: 해당 video의 다양한 augment로부터 얻는 representation / 즉 어떤 변형된 형태의 얼굴이든, 그 핵심 특징은 일치해야 함
        - 다른 instance 비디오: 다른 모든 비디오 instance와의 representation과는 최대한 달라지도록.
### Loss Function
Loss 구성은 다음과 같음.
$$\mathcal{L}=\mathcal{L}_{\textit{distill}}+\mathcal{L}_{\textit{ID}}$$
각각 Multi-scale Landmark Guided Self-distillation Loss, Instance Learning Loss의 합으로 구성.
- **Multi-scale Landmark Guided Self-distillation Loss**
    - student network $g_{\theta_s}$의 output이 teacher network $g_{\theta_t}$의 output과 일치하도록 만드는 loss term
