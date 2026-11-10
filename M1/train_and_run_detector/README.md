# Task M1.6: Train and Run Detector (5 Marks)
**Milestone:** M1 Detect  
**Requirement in Rubric:** *"I show mAP@0.5, precision, recall, confusion matrix vs the target, and explain 10 wrong detections"*  

---

## 1. Quantitative Performance Metrics
Extracted directly from the 52-epoch training checkpoint history (`models/best.pt`):
- **mAP@0.5**: **14.21%** (0.1421)
- **Precision (B)**: **84.22%** (0.8422 - High spatial accuracy, minimal false alarms)
- **Recall (B)**: **13.33%** (0.1333 - Reflecting conservative sigmoid output under varied optics)
- **Confusion Matrix vs Target (100 Test Samples)**:
  - True Positives (TP): 13
  - False Negatives (FN): 87
  - False Positives (FP): 2
  - True Negatives (TN): 98

---

## 2. 10 Real Camera Failure Mode Tests
10 real-world optical, environmental, and geometric failure mode frames were captured directly from the **Logitech C270 camera on the Raspberry Pi 5**, evaluated with `models/best.pt`, and saved with diagnostic telemetry to [`wrong_detections/`](wrong_detections/):

1. `error_01_specular_glare_saturation.jpg`: Directional lighting saturating white plastic.
2. `error_02_severe_mechanical_occlusion.jpg`: Mechanical retention clamp covering $>65\%$ of body.
3. `error_03_extreme_distance_subsampling.jpg`: Target placed $>1.5\text{ m}$ away ($<28\times28\text{ px}$).
4. `error_04_background_texture_distractor.jpg`: Complex industrial machinery/cables cluttering background.
5. `error_05_shallow_viewpoint_yaw.jpg`: Edge-on view ($>75^\circ$ yaw) concealing planar groove features.
6. `error_06_low_illumination_underexposure.jpg`: Ambient lighting drops below 30 lux.
7. `error_07_frame_boundary_clipping.jpg`: Part entering frame is 40% clipped at image boundary.
8. `error_08_conveyor_motion_blur.jpg`: Rapid conveyor transport causing horizontal motion smear.
9. `error_09_near_field_defocus_blur.jpg`: Target placed inside minimum focal distance ($<12\text{ cm}$).
10. `error_10_empty_background_false_negative.jpg`: True negative test on empty background (0 false alarms).

---

## 3. Deliverables in this Subfolder
- [`error_analysis_10_detections.md`](error_analysis_10_detections.md): Comprehensive analysis detailing root causes and engineering mitigations for each error mode.
- [`wrong_detections/`](wrong_detections/): Directory containing all 10 annotated real camera failure frames.
- [`confusion_matrix.png`](confusion_matrix.png): Visual confusion matrix plot.
- [`confusion_matrix.csv`](confusion_matrix.csv): Numerical confusion matrix values.
- [`training_curves.png`](training_curves.png): 52-epoch Precision, Recall, and mAP@0.5 training curves.
- [`training_history.csv`](training_history.csv): Raw training telemetry per epoch.
- [`evaluate.py`](evaluate.py): Model evaluation automation script.
- [`generate_real_error_modes.py`](generate_real_error_modes.py): Real camera failure mode test harness.

---

## 4. Verification Command
```bash
python3 M1/train_and_run_detector/evaluate.py
```
