# ME369P-Final-Project-U07
Repository for Group U07, TEA, final project

High-Level Objective:
Anomaly Detection in Time Series Data (Machine Learning Integration)

Data:
The data set we are using is too large to upload so here is the link to the data instead: https://data.austintexas.gov/Transportation-and-Mobility/Radar-Traffic-Counts/i626-g7ub
For our computers, the dataset was too large to save in excel, so only the first 1,048,575 rows are used in this project
All variable descriptions can be found at the link, however some were excluded from analysis as the proved to not be beneficial

Packages:
SciPy: For data preprocessing, mathematical operations, stats, etc
Numpy: Probably going to need it for other operations
SciKit-learn: For machine learning algorithms and evaluating models/data
Matplotlib: To visualize the results in plots, figures, images, etc

Deliverables:
Module Preprocessing: Process the data and standardize it in preparation for machine learning integration
Machine Learning module: Train a model to detect anomalies in the preprocessed data (Isolation Forest, One-Class SVM, other). Confirm and correct hyperparameters before the learning process begins
Model Evaluation: Evaluate its performance on a dataset and use metrics like precision, recall, or others to detect anomaly
Visualization Module: Create a way to visualize the model's detections against any original time series data
