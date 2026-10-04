import json, re, subprocess
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
rows=[]
n=10485760
for dtype_id,dtype in enumerate([np.float32,np.float16,bfloat16]):
    rng=np.random.default_rng(20261004)
    rng.uniform(1.01,10,n).astype(dtype).tofile('y.bin')
    rng.uniform(-2,2,n).astype(dtype).tofile('dy.bin')
    for repeat in range(3):
        # Rotate order to reduce the association of version with warmup/drift.
        names=['baseline','tile4096','validate']
        names=names[repeat:]+names[:repeat]
        for name in names:
            result=subprocess.run(['./build/'+name,str(dtype_id),str(n),'2'],capture_output=True,text=True,check=True)
            us=float(re.search(r'mean_us=([\d.]+)',result.stdout)[1])
            row=dict(dtype=str(np.dtype(dtype)),variant=name,repeat=repeat,mean_us=us)
            rows.append(row);print(json.dumps(row),flush=True)
Path('benchmark.json').write_text(json.dumps(rows,indent=2))
