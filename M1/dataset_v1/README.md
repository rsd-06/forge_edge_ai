# Task M1.4: Dataset v1 (4 Marks)
**Milestone:** M1 Detect  
**Requirement in Rubric:** *"I label images with tight, correct boxes. I show my own labelled images and the data card"*  

---

## 1. Overview
Documents the custom Roboflow industrial vision dataset utilized to train the YOLOv8 detector:
- Target Object: White injection-molded rectangular electronic sensor enclosure.
- Physical Characteristics: Dual longitudinal mounting grooves, central sensor/LED pinhole aperture, rounded corners.
- Bounding Box Quality: Tight, high-IoU annotations enclosing the physical casing perimeter while excluding background shadows.

---

## 2. Deliverables in this Subfolder
- [`data_card.md`](data_card.md): Comprehensive dataset card detailing source, format, class ontology, physical dimensions, and bounding box tightness audit.
- [`labelled_images/`](labelled_images/): Authentic Roboflow ground-truth training images:
  - `sample_ground_truth_1.jpg`: Enclosure reverse side against high-contrast red industrial fixture.
  - `sample_ground_truth_2.jpg`: Enclosure front face with grooves and pinhole at $45^\circ$ diagonal orientation.
  - `sample_ground_truth_3.jpg`: Enclosure positioned horizontally on dark textured conveyor surface.
