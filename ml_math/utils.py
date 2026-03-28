import math

def sigmoid(x):
    """
    Sigmoid activation function.
    Formula: 1 / (1 + exp(-x))
    """
    return 1 / (1 + math.exp(-x))

def min_max_scaler(data):
    """
    Scales data to a range between 0 and 1.
    Formula: (x - min) / (max - min)
    Works on a 2D list (matrix) where scaling is done per column (feature).
    """
    if not data or not data[0]:
        return data
        
    rows = len(data)
    cols = len(data[0])
    
    # Initialize min and max with the first row
    min_vals = list(data[0])
    max_vals = list(data[0])
    
    # Find min and max for each column
    for i in range(1, rows):
        for j in range(cols):
            if data[i][j] < min_vals[j]:
                min_vals[j] = data[i][j]
            if data[i][j] > max_vals[j]:
                max_vals[j] = data[i][j]
                
    # Scale the data
    scaled_data = []
    for i in range(rows):
        scaled_row = []
        for j in range(cols):
            diff = max_vals[j] - min_vals[j]
            if diff == 0:
                scaled_row.append(0.0)
            else:
                scaled_row.append((data[i][j] - min_vals[j]) / diff)
        scaled_data.append(scaled_row)
        
    return scaled_data

def standard_scaler(data):
    """
    Scales data to have a mean of 0 and a standard deviation of 1.
    Formula: (x - mean) / std
    Works on a 2D list (matrix) where scaling is done per column (feature).
    """
    if not data or not data[0]:
        return data
        
    rows = len(data)
    cols = len(data[0])
    
    # Calculate Mean
    means = [0.0] * cols
    for i in range(rows):
        for j in range(cols):
            means[j] += data[i][j]
    means = [m / rows for m in means]
    
    # Calculate Standard Deviation
    stds = [0.0] * cols
    for i in range(rows):
        for j in range(cols):
            stds[j] += (data[i][j] - means[j]) ** 2
    stds = [math.sqrt(s / rows) for s in stds]
    
    # Scale the data
    scaled_data = []
    for i in range(rows):
        scaled_row = []
        for j in range(cols):
            if stds[j] == 0:
                scaled_row.append(0.0)
            else:
                scaled_row.append((data[i][j] - means[j]) / stds[j])
        scaled_data.append(scaled_row)
        
    return scaled_data
