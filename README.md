# ME369P-Final-Project-U07
### Repository for Group U07, TEA, final project

---

## High-Level Objective
**Anomaly Detection in Time Series Data (Machine Learning Integration)**

---

## Data
- **Dataset Link:** [Radar Traffic Counts](https://data.austintexas.gov/Transportation-and-Mobility/Radar-Traffic-Counts/i626-g7ub)
- **Usage Note:** Due to size constraints, we used only the first 1 million rows for this project. Full variable descriptions are available at the link. The only integer variable we considered as a feature in analysis was Speed. 

---

## Packages
- **SciPy:** Regression analysis and signal noise cancellation
- **Numpy:** Numeric calculations
- **Pandas:** CSV import and data handling
- **SciKit-learn:** Machine learning algorithms and model/data evaluation
- **Matplotlib:** Visualization of results (plots, figures, images, etc.)

---

## Deliverables
1. **Module Preprocessing:** Process and standardize data for machine learning.
2. **Machine Learning Module:** Train a model (Isolation Forest) to detect anomalies. Set and verify hyperparameters before training.
3. **Model Evaluation:** Assess performance using metrics like precision and recall.
4. **Visualization Module:** Visualize model detections against the original time series data.

---

## How to Use Code
### Required Packages
- Install via pip: `numpy`, `scikit-learn`, `pandas`, `scipy`, `matplotlib`

### Steps
1. Download `Radar_Traffic_Counts.csv` or `Test RadarCounts.csv` from this repository, or the full dataset [here](https://data.austintexas.gov/Transportation-and-Mobility/Radar-Traffic-Counts/i626-g7ub). The formers are shortened data files with 20,000 rows each that can be accurately scaled to the full-sized dataset.
   - **Note:** Exporting the full dataset takes about 10-20 minutes. 
2. Run `TEA Learning Module.py`, ensuring the `file_path` is set equal to the appropriate dataset, for example: `file_path = 'Test RadarCounts.csv'`. This script performs data preprocessing, machine learning training and evaluation, and data visualization.
3. **Visualizing Data:** The first bar-graph subplots show average speeds at each hour for the 3 most anomalous intersections. The second subplot illustrates the relationship between car volume and average speed.
   - **Speed Limit Data:** Manually entered for top 3 intersections. For different datasets, this needs to be updated. There was no simple way to extract speed limit data online within the project's time constraints.

---

