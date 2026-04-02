# 🔍 알약 객체 검출 프로젝트
> CODEIT Sprint Project | 2026 AI 9기 | 파트2 2팀

## 📌 프로젝트 개요
알약 이미지에서 객체를 검출하는 AI 모델을 개발하고, 다양한 성능 향상 기법을 적용하여 최적의 검출 성능을 달성하는 프로젝트입니다.

---

## 🏆 최종 결과
| 방법 | mAP@[0.75:0.95] |
|------|----------------|
| Baseline (RT-DETRv4-X) | 0.96833 |
| TTA + WBF + Ensemble | **0.97528** |

---

## 🛠️ 실험 기술
- **Model**: RT-DETRv4-X, YOLO v11
- **Resize**: 640, 1024, etc.
- **Augmentation**: Simple Copy-Paste, Image Inpainting, Mosaic
- **Post-processing**: WBF (Weighted Boxes Fusion)
- **Ensemble**: RT-DETRv4-X + YOLO v11-M
- **TTA**: Horizontal Flip, Vertical Flip, Multi-Scale Resize

---

## 📊 실험 결과 요약

### 모델 비교
| Model | mAP@[0.75:0.95] |
|-------|----------------|
| YOLO v11-N | 0.61426 |
| YOLO v11-M | 0.76016 |
| YOLO v11-X | 0.86357 |
| RT-DETRv4-L | 0.82756 |
| **RT-DETRv4-X** | **0.96833** |

### Data Augmentation
| 방법 | mAP@[0.5:0.95] |
|------|----------------|
| Baseline | 0.9114 |
| Simple Copy-Paste | 0.9861 (+0.0747) |
| Image Inpainting | 0.8921 (-0.0913) |

### 2-Stage Detector
| Detector | Classifier | mAP@[0.75:0.95] |
|----------|------------|----------------|
| RT-DETRv4-L | - (Base) | 0.8276 |
| RT-DETRv4-L | EfficientNet-B4 | 0.9064 |
| RT-DETRv4-L | ConvNeXt | **0.9081** |

## 최종 결과
| Ensemble 1 | Ensemble 2 |  mAP@[0.75:0.95] |
|----------|------------|----------------|
| RT-DETRv4-L(↑) | YOLOv11-M(↓) | **0.9753** |

---

## 👥 팀원 소개
| 이름 | 역할 |
|------|------|
| 김현수 (팀장) | Project Managing, Data Preprocessing, AI Algorithm (AI Research, AI Test) |
| 김영성 | Data Preprocessing, AI Algorithm (AI Research, AI Test) |
| 박도원 | Data Preprocessing, AI Algorithm (AI Research, AI Test) |
| 한성택 | Data Preprocessing, AI Algorithm (AI Research, AI Test) |

---

## 📄 보고서
- [프로젝트 보고서 바로가기](https://drive.google.com/file/d/1OFD0tIfl0BREgKzfmaj97B86ls6bmWWg/view?usp=sharing)
- [협업 일지 - 김영성](https://velog.io/@csd1345/series/%EC%B4%88%EA%B8%89%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80)
- [협업 일지 - 김현수](https://www.notion.so/hyeonsukim/32ef5734f6cd81019835dc6e178b2b95?source=copy_link)
- [협업 일지 - 박도원](https://www.notion.so/325c44689b4380c2a40ad979a2b8109f?source=copy_link)
- [협업 일지 - 한성택](https://www.notion.so/Daily-3345f26f890680869bbbe0ea76648021?source=copy_link)
---
