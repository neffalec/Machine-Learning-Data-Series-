from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#loading and preprocessing the dataset
file_path = r"C:\Users\mayon\Documents\Test RadarCounts.xlsx"
#file_path_shortened = r"C:\Users\mayon\Documents\Radar_Traffic_Counts.xlsx"
data_sample = pd.read_excel(file_path)
data_sample['Read Date'] = pd.to_datetime(data_sample['Read Date'], format='%m/%d/%Y %H:%M', errors='coerce')
categorical_columns = ['Intersection Name', 'Lane', 'Direction']
numerical_columns = ['Volume', 'Occupancy', 'Speed', 'Month', 'Day', 'Year', 'Hour', 'Minute', 'Day of Week']
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(), categorical_columns)
])
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
#apply noise filtering to the 'Speed' data
data_sample['Speed_Smoothed'] = savgol_filter(data_sample['Speed'], window_length=51, polyorder=3)

#APPROACH: WHEN ARE PEOPLE BREAKING THE NORM? (BY THE HOUR)
#calculate average speeds per Intersection-Hour segment
grouped_data = data_sample.groupby(['Intersection Name', 'Hour'])
average_speeds = grouped_data['Speed'].mean().reset_index(name='Average Speed')
count_readings = grouped_data['Speed'].count().reset_index(name='Count')
#join the average speeds with the count of readings
average_speeds = average_speeds.merge(count_readings, on=['Intersection Name', 'Hour'])

#apply Isolation Forest model directly on the average speeds
iso_forest_avg = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.10, 
                                 max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42)
iso_forest_avg.fit(average_speeds[['Average Speed']])  #using only the Average Speed column for anomaly detection
#detect anomalies in average speeds
average_speed_anomalies = iso_forest_avg.predict(average_speeds[['Average Speed']])
#add anomaly flag to the average_speeds DataFrame
average_speeds['Anomaly'] = average_speed_anomalies == -1
#create a mapping from (Intersection Name, Hour) to Anomaly status
anomaly_mapping = dict(zip(average_speeds[['Intersection Name', 'Hour']].apply(tuple, axis=1), average_speeds['Anomaly']))
#apply the mapping to the original data
data_sample['Anomaly'] = data_sample.apply(lambda row: anomaly_mapping.get((row['Intersection Name'], row['Hour']), False), axis=1)

#Output results
print("Total number of anomalous segments detected:", sum(average_speeds['Anomaly']))
#preview anomalous segments with additional details
print("Preview of anomalous segments with additional details:")
anomalous_segments = average_speeds[average_speeds['Anomaly']]
print(anomalous_segments[['Intersection Name', 'Hour', 'Average Speed', 'Count']])

# Count anomalies for each intersection
weighted_anomalies = average_speeds[average_speeds['Anomaly']].groupby('Intersection Name')['Count'].sum()
top_anomalous_intersections = weighted_anomalies.nlargest(3).index

#need to define the speed limits for each of the top intersections
speed_limits = [60, 60, 45]
#set up the matplotlib figure for subplots
fig, axes = plt.subplots(nrows=3, ncols=1)
#meed to define the hour labels for the x-axis
hour_labels = {0: 'Midnight', 4: '4 AM', 8: '8 AM', 12: 'Noon', 16: '4 PM', 20: '8 PM', 23: '11 PM'}

#function for curve fitting (simple linear model)
def linear_model(x, a, b):
    return a * x + b

#iterate over the top anomalous intersections
for i, intersection in enumerate(top_anomalous_intersections):
    #filter data for the current intersection
    intersection_data = data_sample[data_sample['Intersection Name'] == intersection]
    #calculate the average speed for each hour
    average_speed_by_hour = intersection_data.groupby('Hour')['Speed'].mean()
    #plot on the respective subplot
    axes[i].bar(average_speed_by_hour.index, average_speed_by_hour.values, color='blue')
    axes[i].set_title(f'Average Speed by Hour at {intersection}')
    axes[i].set_xlabel('Time of Day')
    axes[i].set_ylabel('Average Speed (mph)')
    #add horizontal line for the speed limit
    axes[i].axhline(y=speed_limits[i], color='red', linestyle='--', label=f'Speed Limit ({speed_limits[i]} mph)')
    axes[i].legend()
    #set the x-axis ticks and labels
    axes[i].set_xticks(list(hour_labels.keys()))
    axes[i].set_xticklabels(list(hour_labels.values()))
plt.tight_layout()
plt.show()

#set up a new matplotlib figure for curve fitting plots
fig, axes = plt.subplots(nrows=3, ncols=1)
#iterate over the top anomalous intersections for curve fitting plots
for i, intersection in enumerate(top_anomalous_intersections):
    #filter data for the current intersection and for anomalies
    intersection_anomalous_data = data_sample[(data_sample['Intersection Name'] == intersection) & (data_sample['Anomaly'])]
    if not intersection_anomalous_data.empty:
        #do the curve fitting
        popt, _ = curve_fit(linear_model, intersection_anomalous_data['Volume'], intersection_anomalous_data['Speed_Smoothed'])
        volume_range = np.linspace(intersection_anomalous_data['Volume'].min(), intersection_anomalous_data['Volume'].max(), 100)
        fitted_speed = linear_model(volume_range, *popt)
        #[plot actual speed data points and the fitted curve on a new subplot
        axes[i].scatter(intersection_anomalous_data['Volume'], intersection_anomalous_data['Speed_Smoothed'], color='green', label='Anomalous Data Points', s=5)
        axes[i].plot(volume_range, fitted_speed, color='orange', label='Fitted Curve')
        axes[i].set_title(f'Curve Fit for Anomalous Data at {intersection}')
        axes[i].set_xlabel('Volume')
        axes[i].set_ylabel('Speed (Smoothed)')
        axes[i].legend()
plt.tight_layout()
plt.show()
