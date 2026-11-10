# Data Card: Industrial Enclosure Vision Dataset v1.0
**Student Name:** Sai Sanjay R (ID: S20)  
**Milestone:** M1 Detect (M1.4 Dataset v1)  
**Platform:** Raspberry Pi 5 (Debian 13 ARM64)  
**Model Checkpoint:** `models/best.pt` (YOLOv8 Nano, 6.17 MB)

---

## 1. Dataset Overview & Physical Object Specification
- **Object Inspected**: Industrial sensor enclosure casing.
- **Physical Characteristics**: White injection-molded rectangular housing featuring:
  - Dual longitudinal mounting groove rails on the front face.
  - A central sensor/LED pinhole aperture.
  - Rounded exterior corners and recessed mating lip.
- **Source**: Roboflow custom dataset annotated by Sai Sanjay R.
- **Target Detection Class**: `0: enclosure` (Single-class object localization).
- **Annotation Format**: Standard YOLO Darknet format (`class_id x_center y_center width height` normalized $[0, 1]$).
- **Input Image Dimensions**: $640 \times 640$ pixels, RGB 3-channel.

---

## 2. Real Ground Truth Evidence & Annotation Quality
- **Annotation Quality**: Ground truth bounding boxes are tightly fitted to the outer physical perimeter of the white casing, omitting background shadows and surface reflections.
- **Surface & Lighting Variations in Dataset**:
  - Sample 1: Plain white enclosure reverse face against high-contrast red industrial fixture.
  - Sample 2: White enclosure front face with sensor pinhole aperture and longitudinal grooves at $45^\circ$ diagonal orientation on red surface.
  - Sample 3: White enclosure horizontal orientation on dark textured industrial conveyor surface.
- **Visual Evidence References**:
  - [Sample Ground Truth 1](file:///home/juice/yolo_inference/M1_Detect/evidence/labelled_images/sample_ground_truth_1.jpg)
  - [Sample Ground Truth 2](file:///home/juice/yolo_inference/M1_Detect/evidence/labelled_images/sample_ground_truth_2.jpg)
  - [Sample Ground Truth 3](file:///home/juice/yolo_inference/M1_Detect/evidence/labelled_images/sample_ground_truth_3.jpg)

---

## 3. Confidence & Calibration Profile
- **Training Convergence**: As extracted from checkpoint training records (52 epochs), the model converged with high precision ($P = 84.2\%$) and low raw confidence distribution ($10\% - 30\%$).
- **Deployment Threshold Recommendation**: When running on the Raspberry Pi 5 under real-world Logitech C270 illumination, a confidence threshold of $\tau = 0.10$ to $0.15$ achieves robust localization without false positives, accurately detecting the white enclosure casing.

---

## 4. Dataset Limitations & Assessment Scope
1. **Single-Class Constraint**: The trained weights identify the presence and spatial boundary of the `enclosure`. They do not distinguish defective vs pristine enclosures.
2. **Illumination & Surface Contrast**: Because the enclosure is matte white, detection is most robust against darker or saturated contrasting backgrounds (e.g. red, black, or grey surfaces) matching the training distribution. Extreme specular reflections on the white plastic can cause confidence attenuation.
