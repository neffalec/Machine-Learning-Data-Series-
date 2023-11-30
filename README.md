# ME369P-Final-Project-U07
Repository for Group U07, TEA, final project

High-Level Objective:
Anomaly Detection in Time Series Data (Machine Learning Integration)

Data:
The data set we are using is too large to upload so here is the link to the data instead: https://data.austintexas.gov/Transportation-and-Mobility/Radar-Traffic-Counts/i626-g7ub
For our computers, the dataset was too large to save in Excel, so only the first 1 million rows were used in this project. All variable descriptions can be found at the link, however, some were excluded from the analysis as they proved to not be beneficial.

Packages:
SciPy: For regression analysis and signal noise cancellation
Numpy: Numeric calculations
Pandas: csv import and data reading
SciKit-learn: For machine learning algorithms and evaluating models/data
Matplotlib: To visualize the results in plots, figures, images, etc

Deliverables:
Module Preprocessing: Process the data and standardize it in preparation for machine learning integration
Machine Learning module: Train a model to detect anomalies in the preprocessed data (Isolation Forest). Confirm and correct hyperparameters before the learning process begins
Model Evaluation: Evaluate its performance on a dataset and use metrics like precision, recall, or others to detect anomaly
Visualization Module: Create a way to visualize the model's detections against any original time series data

HOW TO USE CODE:

Packages to download (all through pip): numpy, scikit-learn, pandas, scipy, and matplotlib 
1. Download 'Radar_Traffic_Counts.csv' and 'Test RadarCounts.csv' files, these will be shortened data files with 20,000 rows. (Full dataset can be downloaded at https://data.austintexas.gov/Transportation-and-Mobility/Radar-Traffic-Counts/i626-g7ub, by clicking export -> download. Exporting takes ~10-20 minutes due to their servers, and we can't upload these files to GitHub so the shortened file is a way to show our code works without waiting awhile for a download. The first file is used for preprocessing and the second file is used for analysis. Both are needed.
2. Run 'TEA Learning Module.py' file, making sure that file_path is set to 'Test RadarCounts.csv' This script performs data preprocessing in portion 1, machine learning training and evaluation in portion 2, and data visualization in portion 3.
3. Visualizing the data: The first bar-graph subplots represent the average speeds driven at each hour of the day for the 3 most anomalous intersections. The second subplot shows a relationship between the volume of cars on the road versus the average speed traveled for that volume. **Note:** Speed limit data for top 3 intersections was manually entered. For different data sets this would need to be changed, but there was no simple way to scrub speed limit data for each intersection online given the time limits.
