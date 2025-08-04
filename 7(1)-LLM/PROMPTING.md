# 7-(1) LLM 과제 보고서

## 1. 정답률 비교

| Prompting Type   | 0-shot | 3-shot | 5-shot |
| ---------------- | ------ | ------ | ------ |
| Direct Prompting | 0.18   | 0.18   | 0.22   |
| CoT Prompting    | 0.64   | 0.68   | 0.6    |
| My Prompting     | 0.78   | 0.74   | 0.8    |

---

## 2. CoT Prompting이 Direct Prompting보다 좋은 이유

### 2.1. 문제 해결 과정에서 사고 흐름을 지정해준다.

- Direct Prompting은 질문에 대한 정답을 바로 요구하기 때문에 모델 내부에서 사고 과정을 추측하게 됨
- CoT Prompting은 계산 과정의 흐름을 따라가도록 요구하기 때문에 모델이 사고 흐름에 따라 답을 구할 수 있음

### 2.2. 복잡한 연산을 나누어 처리하게 됨

- Direct Prompting 방식과 달리, CoT Prompting에서는 여러 단계의 사고가 필요한 문제를 나눠서 풀 수 있으므로 오류 가능성을 줄일 수 있음

## 3. My Prompting이 CoT Prompting보다 좋은 이유

### 3.1. 출력 형식에 대한 제한 강화

- My Prompting에서는 CoT Prompting에 비해 더 강력히 출력 형식을 제한 하였음
  - 'Answer: 숫자만' 형태로 출력되게 하였으며, 단위를 제거하는 등의 강력하게 제한함
- 사고 흐름을 구조화 해 계산 실수를 줄임
  - 수치 추출, 단계별 계산, 마지막에 숫자만 정답을 출력하도록 하여 계산 실수 및 형식 오류를 줄일 수 있음 -> 모델의 일관성이 올라감
