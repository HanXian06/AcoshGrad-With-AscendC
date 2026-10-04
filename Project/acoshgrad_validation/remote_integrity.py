import hashlib, json
from pathlib import Path
root=Path('/mnt/workspace/acoshgrad_1916_20261004/acoshgrad_problem_1916_template')
names=['kernel.asc','main.asc','CMakeLists.txt','run.sh','data_utils.h','scripts/AcoshGrad.py','scripts/gen_data.py','scripts/verify_result.py']
print(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},indent=2))
