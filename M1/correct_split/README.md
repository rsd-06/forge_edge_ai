# Task M1.5: Correct Split (2 Marks)
**Milestone:** M1 Detect  
**Requirement in Rubric:** *"I explain and show the 70/20/10 split with no leakage"*  

---

## 1. Overview
Validates the statistical dataset partition and verifies the total absence of data leakage across experimental splits:
- **70% Training Partition**: Supervised weight optimization.
- **20% Validation Partition**: Hyperparameter selection and early-stopping checkpointing.
- **10% Testing Partition**: Held-out unseen benchmark set.
- **Data Leakage Check**: SHA-256 cryptographic hashing performed across all images to ensure no identical or near-identical frames exist across multiple partitions.

---

## 2. Deliverables in this Subfolder
- [`split_report.md`](split_report.md): Formal dataset split report detailing partition counts, class distributions, cryptographic deduplication methodology, and zero data leakage proof.
