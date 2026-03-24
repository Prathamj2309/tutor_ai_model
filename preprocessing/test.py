from datasets import load_dataset

import pandas as pd
import numpy as np

# Load January session evaluation set
jan_data = load_dataset("PhysicsWallahAI/JEE-Main-2025-Math", "jan", split="test")

# Load April session evaluation set
apr_data = load_dataset("PhysicsWallahAI/JEE-Main-2025-Math", "apr", split="test")

jan_data = jan_data.to_pandas()
apr_data = apr_data.to_pandas()

df = pd.concat([jan_data, apr_data], ignore_index = True)

df.to_csv("jee_math.csv", index = False)
