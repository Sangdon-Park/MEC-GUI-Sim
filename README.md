# MEC Sim Enhanced with Visualized Techniques  
도시 규모의 MEC(Multi-access Edge Computing) 시뮬레이션을 **2D 격자 기반 ISO 투영** 그래픽으로 표현하고,  
Edge/Cloud 리소스 할당, 네트워크 품질 변동(Rayleigh Fading, Shadowing, Path Loss) 등을 시뮬레이션하여  
메타버스 관점에서 다양한 정책 시뮬레이션이 가능하도록 구현한 종합 플랫폼입니다.

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [주요 기능](#주요-기능)  
3. [시스템 요구사항](#시스템-요구사항)  
4. [설치 및 실행 방법](#설치-및-실행-방법)  
5. [코드 구조](#코드-구조)  
    - [1) 전역 변수 및 상수](#1-전역-변수-및-상수)  
    - [2) 핵심 클래스](#2-핵심-클래스)  
    - [3) 주요 함수 및 로직](#3-주요-함수-및-로직)  
6. [사용 방법 안내 (단축키)](#사용-방법-안내-단축키)  
7. [시뮬레이션 데이터 수집 및 결과 확인](#시뮬레이션-데이터-수집-및-결과-확인)  
8. [확장 포인트 및 응용](#확장-포인트-및-응용)  
9. [라이선스](#라이선스)  

---

## 프로젝트 개요
이 코드는 **MEC(Multi-access Edge Computing)** 환경을 도시 단위로 모델링하여, 시뮬레이션 및 시각화를 수행하는 종합 툴입니다.  
다음과 같은 연구/개발 과제에 활용 가능합니다.
- Edge/Cloud 리소스 동적 할당 및 가격 정책 연구
- 사용자 이동성(모빌리티) 기반의 네트워크 품질 변화 관찰
- Rayleigh 페이딩, Path Loss, Shadowing 등을 포함한 무선 채널 특성 모사
- Federated Learning, Split Computing, Personalization, Early Exit 등 다양한 정책의 효과 검증  
- 시뮬레이션 결과를 그래프와 CSV로 저장하여 추후 데이터 분석에 활용  

코드는 **도시의 2D 격자**를 기반으로, **Isometric(ISO) 투영** 기법을 사용해 빌딩, 나무, 기지국, Edge 서버 등이  
약간의 3D 느낌을 주도록 그려집니다.

---

## 주요 기능
1. **ISO 투영 기반 시각화**  
   - 2D 격자 좌표를 ISO 좌표로 변환, 마치 등각 투영된 3D 도시처럼 표현  
   - 빌딩, 나무, 기지국, Edge 서버 등의 요소 시각화  

2. **MEC & Cloud 자원 동적 관리**  
   - 사용자(노드)가 Local(자체 처리), Edge, Cloud 중 어느 곳에 자원을 임대할지 의사결정  
   - 가격(단가)와 QoS(속도) 등에 따른 수요-공급 연계 모델(수요-가격 탄력성, Marginal Cost 계산 등)  
   - Edge 서버별 임계 Queue(EDGE_SERVER_MAX_Q), Cloud Queue(CLOUD_MAX_Q) 관리  

3. **무선채널 모델링**  
   - Path Loss, Shadowing, Rayleigh Fading 등을 통해 Down/Uplink가 실시간 변동  
   - LTE, 5G 등 통신 모드 임의 할당 및 페이딩 주기(FADING_UPDATE_INTERVAL)마다 링크 재계산  

4. **시뮬레이션 인터랙션**  
   - **Pause(일시정지) 모드**에서 링크 상세 정보를 확인 (페이딩, 업/다운링크 속도, Path Loss, Shadowing, Rayleigh 계수 등)  
   - 자원(Edge 서버) 삭제/추가, 기지국 생성/삭제 가능  

5. **데이터 로그 및 결과 시각화**  
   - CSV로 실험 결과 저장  
   - matplotlib으로 자동 그래프 생성(수익, 사용자 수, 태스크 유형 분포, 평균 링크 속도 등)  
   - 실행 종료 시 그래프 창(plt.show())과 PNG 파일이 함께 생성  

---

## 시스템 요구사항
- **Python 3.7 이상**  
- `pygame` (2.x 이상 권장)  
- `numpy`  
- `matplotlib`  
- `scipy`  
- `pandas` (결과 CSV 생성 시 사용)

설치 예시:
```bash
pip install pygame numpy matplotlib scipy pandas
```

---

## 설치 및 실행 방법
1. 본 저장소를 클론 또는 다운로드  
2. `main()` 함수가 포함된 스크립트(`mec_sim.py` 등) 위치에서 다음 명령어 실행:
    ```bash
    python mec_sim.py
    ```
3. 시뮬레이션 창이 뜨면, 다음 단축키를 활용하여 시뮬레이션을 제어할 수 있습니다.  

---

## 코드 구조

### 1) 전역 변수 및 상수
- **PAUSED**, **SIMULATION_SPEED** 등은 시뮬레이션 속도와 일시정지 여부를 결정  
- **GRID_W**, **GRID_H**: 격자 크기  
- **BLOCK_SIZE**: 하나의 격자 블록이 실제(화면)에서 차지하는 픽셀 단위 길이  
- **ISO_SCALE**, **ISO_OFFSETX**, **ISO_OFFSETY**: ISO 투영 스케일 및 화면 좌표 오프셋  
- **EDGE_SERVERS_COUNT**, **EDGE_SERVER_MAX_Q**: Edge 서버 개수 및 서버별 큐 최대 용량  
- **CLOUD_MAX_Q**: Cloud(백엔드) 큐 최대 용량  
- **RAYLEIGH_SCALE**, **FADING_UPDATE_INTERVAL**: 페이딩 파라미터  
- **START_HOUR**, **START_MINUTE**: 시뮬레이션 시작 시간  

그 외 다양한 파라미터(가격, fading, 건물/트리 개수 등)가 코드 상단에 정의되어 있습니다.

### 2) 핵심 클래스
1. **`Person`(pygame.sprite.Sprite)**
   - 시뮬레이션상의 개별 사용자(노드)를 나타냄  
   - 이동(목적지를 랜덤 선정하여 BFS 경로 탐색 후 이동), 서비스 계약(Edge/Cloud/Local) 로직 포함  
   - 링크 속도(Rayleigh Fading, Shadowing, Path Loss 등) 주기적 재계산  
   - **apply_early_exit**, **apply_federated**, **apply_split**, **apply_personalization** 등 사용 기술 플래그 저장  

2. **`BuildingOccupant`**
   - 빌딩 내부에 있는 사용자(노출되지 않음)를 간단 모델  
   - 일정 시간이 지나면 빌딩에서 나와서 실제 `Person` 오브젝트로 전환됨  

3. **`MECEnvironment`**
   - 전체 시뮬레이션 환경을 관리하는 컨트롤러 클래스  
   - 맵(빌딩, 트리, 기지국, Edge 서버) 정보 및 `Person`(사용자) 객체들을 관리  
   - **step()**에서 1분 단위(또는 SIMULATION_SPEED 배속)에 따라 사용자 스폰, 계약 로직, 가격 재계산 등을 진행  
   - **draw()**에서 화면에 도시, 서버, 사람, 링크, 로그 패널 등을 그려줌  
   - **data** 구조체를 통해 시뮬레이션 결과(수익, 사용자 수, 평균 속도 등)를 주기적으로 수집 -> CSV 저장, 그래프 생성  

4. **`Menu`**  
   - UI에서 Edge 서버의 오퍼레이터 변경, 추가/삭제, 기지국 추가/삭제 등을 처리하기 위한 간단한 메뉴 객체  

### 3) 주요 함수 및 로직
- **`to_iso(x, y)`**  
  2D 격자 좌표(x, y)를 ISO 투영 좌표로 변환하는 함수  
- **`bfs_path(start, goal, blocked)`**  
  BFS를 사용하여 `start`~`goal` 간 최단 경로(격자 단위)를 리스트로 반환  
- **`demand_price(q)`, `cost_function(q, N)`, `compute_price(Q, q, N)`**  
  - 가격 수요곡선, 비용함수, 단위가격 등을 계산하는 핵심 로직  
  - (선형/비선형) 수요-공급 이론에 기반한 마진, 이윤, 탄력성 계산  
- **`Person.update()`**  
  - (1) 남은 lifetime 감소  
  - (2) Edge/Cloud 계약 처리 및 만료 시 정리  
  - (3) Rayleigh 페이딩 및 Path Loss 재계산 -> 링크 속도 업데이트  
- **`MECEnvironment.step()`**  
  - (1) 시간 진행 -> 도착률(Poisson)에 따라 새 사용자(people) 스폰, 빌딩 내부 occupant 스폰  
  - (2) 사용자/Occupant/Cloud 계약 상태 업데이트  
  - (3) 정해진 간격마다 Edge 가격 재계산, 데이터 수집  
- **`MECEnvironment.draw()`**  
  - ISO 투영된 맵, 도시 객체, 서버, 사용자, 링크 연결선 그리기  
  - 상단 패널(현재 시간/수익/기술 적용 현황), 우측 패널(Edge 서버 정보), Log 패널(이벤트 로그) 표시  

---

## 사용 방법 안내 (단축키)
- **Space**: 시뮬레이션 일시정지(PAUSED) 토글  
- **숫자키(1~8)**: 시뮬레이션 속도 배율 변경 (1x ~ 128x)  
- **C**: 일시정지 상태에서 연결 상세 정보 표시 토글 (SHOW_CONNECTION_DETAILS)  
- **W, A, S, D**: ISO 맵 시점 이동(카메라 움직임)  
- **마우스 휠**: ISO 스케일 확대/축소  

일시정지 상태에서 사용자 링크를 클릭하면, 클릭된 링크(Edge or Cloud)에 대한 Fading/Path Loss 정보가 팝업으로 표시됩니다.

---

## 시뮬레이션 데이터 수집 및 결과 확인
- **시뮬레이션 도중**  
  - 상단 정보 패널: 현재 시간, 전체 사용자 수, 수익, 기술 적용(FL, Split 등) 사용자 카운트 표시  
  - 우측 패널: 각 Edge 서버(Queue, 수익, 운영사 등) 표시  
  - 하단 Log 패널: 계약 체결/해지, Edge 서버 추가/삭제 등 주요 이벤트 로그  

- **시뮬레이션 종료 시**  
  - `MECEnvironment.save_and_plot_data()` 호출을 통해 수집된 데이터(`df.to_csv(...)`)를 CSV로 저장  
  - matplotlib로 그래프(9개 서브플롯) 생성 -> PNG 파일로도 자동 저장  
  - 그래프 창이 팝업으로 뜨므로, 종료하거나 저장해서 사용 가능  

CSV 및 PNG는 `results/` 폴더에 날짜/시간 정보를 붙여 자동 저장됩니다.

---

## 확장 포인트 및 응용
1. **추가적인 서비스 정책(FL, Personalization) 구현**  
   - 각 `Person`에 새로운 속성(학습 파라미터, 모델 크기 등)을 추가하고, ‘apply_federated’를 세부 로직으로 확장  
2. **다양한 네트워크 모델링**  
   - 현재는 Rayleigh, Shadowing, Path Loss를 단순 합산 방식으로 반영  
   - 5G mmWave 특성(LoS/NLoS, 트래픽 밀집, Beamforming 등)을 추가로 고려 가능  
3. **멀티 Edge 서버 간 협력/Offloading**  
   - 현재 코드는 사용자-단일 Edge 간 계약 구조이지만, Edge-Edge 간 동적 offloading 및 Joint-Resource Allocation 정책도 확장 가능  
4. **실제 도시 맵 데이터 적용**  
   - 현재는 랜덤 배치로 구성된 빌딩, 트리 등이므로, 실제 지도와 연동하여 격자 구조를 생성 가능  
5. **대규모 사용자 시뮬레이션**  
   - Poisson 도착률, Exponential 분포 등은 샘플 예시이므로, 인구 통계 기반(출퇴근, 주말 패턴 등)으로 확장 가능  

---

## 라이선스
이 프로젝트는 자유롭게 연구용/교육용 목적으로 사용 및 수정 가능합니다.  
다만, 외부 라이브러리(pygame, numpy, matplotlib, scipy, pandas)는 각 라이브러리의 라이선스에 따릅니다.  
추가 문의나 공유는 언제든지 환영합니다.

---

## Acknowledgement

이 프로젝트는 **기초연구사업 - 신진연구 - 세종과학펠로우십(국내트랙)** 과제를 통해 지원을 받아 진행 중인 연구 결과물입니다.

- **국문 과제명**: VR, AR 등 미래 산업 환경에서 엣지 컴퓨팅 시뮬레이터를 활용한 학습 기반의 엣지 컴퓨팅 최적 운영 기법 개발 연구  
- **영문 과제명**: Research on the development of optimal operating techniques for edge computing based on learning using edge computing simulators in future industrial environments such as VR/AR

**This work was supported by the National Research Foundation of Korea (NRF) Grant funded by Korea Government [Ministry of Science and ICT (MSIT)] under Grant 2022R1C1C2007724.**
