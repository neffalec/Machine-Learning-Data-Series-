## Final ME369P
## Names: Alec Neff and Thomas Veenstra
## Group : TEA (U07)

import csv
import numpy as np
import scipy as sp

# Read data from excel using the CSV reader
raw_data = open('Radar_Traffic_Counts.csv')
reader = csv.reader(raw_data)
speed_data_list = []
# Skip the header for ease of data processing
next(reader)
for row in reader:
    speed_data_list.append(row)
raw_data.close()

# Convert data to a useable format, in this case an array from Numpy
speed_data = np.array(speed_data_list)
# Removing data not necessary for our analysis
trimmed_data = np.delete(speed_data, [0, 1, 2, 3, 7, 15], 1)
# Reordering the data so string valued columns are first
# This will make splitting the data much easier
trimmed_data = trimmed_data[:, [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]]
location_direction_data, speed_time_data, leftover = np.hsplit(trimmed_data, [3, 11])
# Columns of location_direction_data are as follows:
# Intersection Name, Lane, and Direction

# Columns of speed_time_data are as follows:
# Volume, Speed, Month, Day, Year, Hour, Minute, and Day of Week

# leftover is an empty array

# Convert speed and time data to ints
# Location and direction data will remain as strings
speed_time_data = speed_time_data.astype(int)

# Split both data sets in two for training and validation sets
location_training_set, location_validation_set, leftover = np.vsplit(location_direction_data, [524287, 1048575])
speed_training_set, speed_validation_set, leftover = np.vsplit(speed_time_data, [524287, 1048575])
