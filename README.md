# Variations in albumin concentration correspond to changes in hemoglobin, indicating a common underlying physiological regulation during chronic inflammation.

This repository contains a project analyzing a sparse, longitudinal biomarker dataset. The project documents an analytical journey from an initial, sparse-data hypothesis (Calprotectin vs. Hemoglobin) to a robust, data-dense finding: a significant concurrent correlation between Albumin and Hemoglobin.

The analysis demonstrates methods for handling statistical artifacts, data sparsity, and biological confounders, culminating in a validated, significant finding.

![Main Finding: Albumin vs. Hemoglobin Correlation](Albumin_vs_Hemoglobin_Scatter.png)

---

## The Analytical Journey: From Sparsity to Significance

The core of this project is the pivot from a compelling but statistically untestable hypothesis to a provable, data-driven one.

### Phase 1: The Initial (Sparse) Hypothesis: Calprotectin vs. Hemoglobin

The project began by attempting to validate the medical literature's link between gut inflammation (Fecal Calprotectin) and anemia (Hemoglobin).

* **Initial Artifact:** A cross-correlation on *interpolated* data produced a "perfect" $p=0$ result. This was identified as a **statistical artifact** (a "bad $p=0$") caused by correlating two "connect-the-dots" lines, not real data.
* **Honest Analysis:** A robust `merge_asof` analysis on the *raw*, non-interpolated data found the true, underlying relationship:
    * **Correlation (r):** -0.120 (Directionally correct)
    * **P-value (p):** 0.164 (Statistically insignificant)
* **Conclusion:** The initial hypothesis was not wrong, but it was **unprovable** with this dataset. The core problem was **Data Sparsity**—stool tests (Calprotectin) and blood tests (Hemoglobin) are rarely collected at the same time, leading to a weak signal ($r = -0.12$) and low power ($n = 137$).

---

## The Pivot & Key Finding: A "Dense" Analysis

The analysis pivoted to a "dense" hypothesis: analyzing markers collected from the **same blood panel**, thus eliminating the sparsity problem.

### Finding 1: A Strong Concurrent Correlation (Albumin vs. Hemoglobin)

The primary finding is a strong, significant link between Albumin and Hemoglobin. A sensitivity analysis showed this link is **concurrent** (strongest at a 1-day window) and not temporally predictive.

* **Statistical Finding:**
    * **Correlation (r):** +0.551
    * **P-value (p):** < 0.001 (full value: $6.14 \times 10^{-12}$)
    * **Sample Size (n):** 133 (real, concurrent data pairs)

* **Biological Interpretation:** The positive correlation is biologically correct.
    * **Healthy State:** High Albumin (good nutrition/low inflammation) is paired with High Hemoglobin (no anemia).
    * **Inflamed State:** Low Albumin (malnutrition/high inflammation) is paired with Low Hemoglobin (anemia).

* **Visual Validation:** A linear regression scatter plot (see `Albumin_vs_Hemoglobin_Scatter.png`) was generated, confirming the relationship is linear and not driven by outliers.

---

## How to Run

This analysis was conducted in Python using Jupyter Notebooks.

### Dependencies

* pandas
* numpy
* scipy
* matplotlib
* seaborn

You can install all dependencies via:
```bash
pip install pandas numpy scipy matplotlib seaborn
