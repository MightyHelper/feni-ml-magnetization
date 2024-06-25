import re
from pathlib import Path
from pandas import DataFrame
algorithms = ["ranger", "catboost", "svm", "glmnet"]
splits = ["60-40", "70-30", "no-split"]


pattern = re.compile(r"""
    (?P<user_time>\d+\.\d+)user\s+             # user time
    (?P<system_time>\d+\.\d+)system\s+         # system time
    (?P<elapsed_time>\d+:\d+\.\d+)elapsed\s+   # elapsed time
    (?P<cpu_usage>\d+)%CPU\s+                  # CPU usage
    \(0avgtext\+0avgdata\s+(?P<max_resident>\d+)maxresident\)k\s+  # max resident size
    (?P<inputs>\d+)inputs\+(?P<outputs>\d+)outputs\s+              # inputs and outputs
    \((?P<major_faults>\d+)major\+(?P<minor_faults>\d+)minor\)pagefaults\s+  # page faults
    (?P<swaps>\d+)swaps                       # swaps
""", re.VERBOSE)


items = []

for split in splits:
    for algorithm in algorithms:
        file = Path(__file__).parent.parent / split / "FeNiMl" / (algorithm + ".time")
        text = file.read_text()
        # Extract data
        match = pattern.search(text)
        if match:
            extracted_data = match.groupdict()
            extracted_data.update(algorithm=algorithm, split=split)
            items.append(extracted_data)
        else:
            print("No match found")

df = DataFrame(items)
df.to_csv("time_spent.csv", index=False)
print(df)