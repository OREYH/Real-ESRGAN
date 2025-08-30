# Super-Resolution Pipeline for Oral Images

이 파이프라인은 논문에서 설명된 방법론을 구현합니다:

## 파이프라인 순서
1. **입력 검증**: 256×256 구강 이미지 (정상/충치) 준비 및 검증
2. **6배 업스케일링**: Pre-trained Real-ESRGAN x4plus 모델로 1536×1536 생성
3. **최종 리사이즈**: Lanczos 보간법으로 1920×1080 최종 해상도 변환

## 주요 특징
- **모델**: Pre-trained realesrgan-x4plus (파인튜닝 없음)
- **입력**: 256×256 해상도 자동 검증/변환
- **출력**: 1920×1080 고품질 임상용 이미지
- **멀티프로세싱**: 병렬 처리로 성능 최적화

## 사용법

### 기본 사용
```python
from super_resolution_pipeline import SuperResolutionPipeline

# 파이프라인 초기화
pipeline = SuperResolutionPipeline(
    input_dir="inputs/11월 18일",        # 입력 이미지 폴더
    output_dir="outputs/final_1920x1080" # 출력 폴더
)

# 전체 파이프라인 실행
output_files = pipeline.run_pipeline()
```

### 커맨드라인 실행
```bash
python super_resolution_pipeline.py
```

## 파일 구조
```
inputs/11월 18일/           # 256×256 입력 이미지들
├── image1.png
├── image2.png
└── ...

outputs/final_1920x1080/    # 1920×1080 최종 결과
├── image1.png
├── image2.png
└── ...
```

## 기술적 세부사항

### Real-ESRGAN 설정
- **모델명**: RealESRGAN_x4plus
- **배율**: 6× (256×256 → 1536×1536)
- **정밀도**: FP32 (안정성)
- **파인튜닝**: 없음 (pre-trained 모델 직접 사용)

### 리사이즈 설정
- **알고리즘**: Lanczos 보간법
- **최종 해상도**: 1920×1080 (FHD)
- **품질**: 임상 사용 적합한 고품질

## 논문 방법론 준수
본 구현은 다음 논문 내용을 정확히 따릅니다:

> "For super-resolution, we used the 256×256 generated oral images (healthy and decayed) as input. The pre-trained realesrgan-x4plus model was applied directly, without further fine-tuning and using the default parameters, to produce super-resolution images of 1536×1536 resolution (6× upscaling). Subsequently, these images were resized to a final resolution of 1920×1080 using Lanczos interpolation, achieving high-quality outputs appropriate for clinical use."
