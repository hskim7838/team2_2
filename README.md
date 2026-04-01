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

## 🛠️ 사용 기술
- **Model**: RT-DETRv4-X, YOLO v11
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
[프로젝트 보고서 바로가기]([URL](https://drive.google.com/file/d/1OFD0tIfl0BREgKzfmaj97B86ls6bmWWg/view?usp=sharing))
[팀 스페이스 (협업 일지) 바로가기]([URL]([https://drive.google.com/file/d/1OFD0tIfl0BREgKzfmaj97B86ls6bmWWg/view?usp=sharing](https://www.notion.so/hyeonsukim/2-2-32ef5734f6cd8046a9e9db90f442b8de?source=copy_link)))
---
