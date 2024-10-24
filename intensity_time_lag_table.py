import os
import json
import pandas as pd

# Define the combinations of intensity and time_lag
intensity_values = ["real", "0", "19.3"]
time_lag_values = ["real", "0", "0.212"]

# Base output directory
base_output_dir = "output"

# Prepare an empty dictionary to store mAPs
results = {}

# Iterate through each combination of intensity and time_lag
for intensity in intensity_values:
    results[intensity] = {}
    for time_lag in time_lag_values:
        # Define the directory name based on the combination
        result_dir = f"{base_output_dir}/intensity_{intensity}_time_lag_{time_lag}"
        metrics_file = os.path.join(result_dir, "metrics_summary.json")

        try:
            # Read the metrics_summary.json file and extract mean_ap
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                mean_ap = metrics.get("mean_ap", "N/A")
        except FileNotFoundError:
            mean_ap = "N/A"

        # Store the mAP value in the results dictionary
        results[intensity][time_lag] = mean_ap

# Convert results to a DataFrame for better formatting
df = pd.DataFrame(results)
df.index.name = "time_lag"
df.columns.name = "intensity"

# Print the table of mAPs
print(df)
