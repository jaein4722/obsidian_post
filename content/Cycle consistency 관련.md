아이디어 3. Cycle Consistency Distillation for Extreme Quantization
1. 개념 요약
- 매우 극단적인 양자화(2비트 이하)에서, teacher → student 증류만으론 부족할 때, Cycle Consistency라는 개념을 결합.
- 예: teacher output을 기반으로 student가 예측하는 임시 z를 만든 뒤, 그 z를 다시 teacher에게 인풋으로 넣어 “재구성”한 결과가 원본 teacher output과 유사하도록 제약
- 마치 GAN의 CycleGAN이 “A→B 변환 후 B→A로 되돌렸을 때 원본과 동일”하다는 속성을 쓰듯이, teacher ↔ student 변환 간 cycle-consistency 손실을 부여해, 정보 손실을 줄이고자 함.  

2. 구현 가능 시나리오
	1. Teacher Output = high dimensional feature, Student Output = low-bit feature.
	2. Forward cycle: teacher → (projection) → student representation.
	3. Backward cycle: student representation → (inverse projection) → teacher representation.
	4. Cycle loss: 최종 teacher representation이 원본 teacher representation과 L2/L1 로 유사하도록.
이는 student 모델이 단순히 teacher의 로짓만 모방하는 게 아니라, teacher의 표현 공간을 양자화된 형태로 재현했다가 다시 teacher 공간으로 복원 가능한 수준으로 학습을 강제함.

3. 기대 효과
- 극저비트(1~2비트) 학습 시 발생하는 “비가역적 정보 손실”이 줄어들 수 있음.
- teacher–student 표현 사이를 왕복(cycle)하며 “유효 정보”를 최대한 유지하도록 유도.

4. 단점 및 고려 사항
- 추가 모듈(inverse transform)이 필요 → 학습 복잡도 증가.
- inference 시에는 inverse transform이 필요 없지만, 학습 시에는 teacher와 student 외에 cycle consistency 모듈이 있어야 함.
- 학습 안정화가 어려울 수 있음(마치 GAN 학습처럼) → loss 스케줄링, warm-up이 필요.

![[Pasted image 20250710170937.png]]
1. CycleGAN에서 제시된 cycle consistency 개념 차용. teacher와 student 간 cycle consistency를 부여해 student가 더 teacher의 feature 공간을 잘 학습하도록 만드는 것. teacher의 output을 기반으로 student가 어떤 예측 결과 z를 만들면, 그 z를 다시 teacher의 input으로 넣어 reconstruction 수행, 그 결과가 원본 teacher output과 똑같도록 모델을 학습시키는 아이디어.
2. 의의: 단순 distillation만을 수행했을때보다 student가 teacher의 표현 공간을 훨씬 더 잘 학습할 것이라고 생각. 원본 정보를 조금 더 잘 보존할 수 있지 않을까? 일종의 조금 더 심화된 distillation이라고 할 수 있으려나? 사실 아이디어 자체는 teacher와 student 간의 정보를 교환할 때, 단순 정보만을 넘겨주는 것이 아니라 비대칭 encoder-decoder를 사용해서 student의 구조에 맞게 teacher 결과를 변환하자는 아이디어에서 떠올림.
3. 기존에 이미 있나?: distillation의 측면에서 cycle consistency를 도입한 연구는 내 기준 찾지 못함. 굳이 있다면 기존 CycleGAN 연구가 있을 수 있겠는데, 얘는 mode-collapse 문제를 해결하고자 cycle consistency를 도입한 것이지, distillation의 측면과는 완전히 방향성이 다름. teacher와 student 간의 cycle consistency에 관해 다룬 논문이 있다면 물론 이것도 완전 새로운 방향성이 아님.
4. cycle consistency를 어떻게 다른 task에 적용하는가?: feature distillation처럼 진행하되, output을 기준으로 결정한다던가.. 이게 좀 더 예민할 것 같긴한데… 결국은 feature dilstillation과 동일해보임.
5. 아니면 그냥 sem seg 모델의 특정 layer 에서의 img to img model을 만들어보는 건 어떤가?
6.  제안된 방법을 읽어보았을 때, 결국 teacher의 input과 output의 차원이 같아야 가능한 방법인 것 같다. 그렇다면 생성모델을 제외한 모델에서의 사용처가 떠오르지 않으니, 다른 사용처를 생각해보거나, 아니면 정말로 autoencoder 등의 모델을 대상으로 수행해야할 것 같다.
7. 지금 제안한 방법은 teacher -> student -> teacher의 reconstruction에 집중한 것으로 보이는데, CycleGAN에서의 cycle consistency를 더욱 참조하여, 역방향 loss에 대한 고려는 어떠한가? (student -> teacher -> student)