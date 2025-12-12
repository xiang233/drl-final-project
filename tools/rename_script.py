
import os
from pathlib import Path

DIR = Path("generated_data")

if not DIR.exists():
    raise SystemExit(f"{DIR} does not exist")

for path in sorted(DIR.glob("*pessimist*")):
    new_name = path.name.replace("pessimist", "alpha")
    new_path = path.with_name(new_name)
    print(f"{path}  ->  {new_path}")
    path.rename(new_path)
