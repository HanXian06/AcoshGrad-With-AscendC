import json
import subprocess
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

results = []
for dtype_id, dtype in enumerate([np.float32, np.float16, bfloat16]):
    sizes = [1, 7, 17, 2047, 2048, 2049, 65537, 163839, 163840, 163841, 262151, 1048576, 10485760]
    for n in sizes:
        rng = np.random.default_rng(1916+n)
        y = rng.uniform(1.01, 10, n).astype(dtype)
        dy = rng.uniform(-2, 2, n).astype(dtype)
        # Include the smallest legal values after conversion in every case.
        y[:min(n,5)] = np.array([1.01,1.02,1.1,2,10][:min(n,5)],dtype=dtype)
        y.tofile('y.bin'); dy.tofile('dy.bin')
        yf=y.astype(np.float32); df=dy.astype(np.float32)
        golden=(df / np.sqrt(yf*yf-np.float32(1))).astype(dtype).astype(np.float32)
        rank=0 if n==1 else (8 if n==65537 else 2)
        run=subprocess.run(['./build/validate',str(dtype_id),str(n),str(rank)],capture_output=True,text=True,check=True)
        actual=np.fromfile('z.bin',dtype=dtype).astype(np.float32)
        atol,rtol,limit=[(9.77e-4,1.53e-5,1e-2),(1.95e-3,1.95e-3,1e-1),(1.56e-2,1.56e-2,1)][dtype_id]
        err=np.abs(actual-golden)
        ratio=float(np.mean(err<=atol+rtol*np.abs(golden)))
        maxerr=float(np.max(err))
        passed=ratio>=.99 and maxerr<=limit and np.all(np.isfinite(actual))
        row=dict(dtype=str(np.dtype(dtype)),n=n,rank=rank,matched_ratio=ratio,max_abs_error=maxerr,passed=bool(passed),timing=run.stdout.strip())
        results.append(row); print(json.dumps(row),flush=True)
        assert passed, row
    # Boundary behavior is checked separately from finite tolerance comparison.
    y=np.array([1,1,1,.5,2,1.01,10],dtype=dtype)
    dy=np.array([1,-1,0,1,0,-2,2],dtype=dtype)
    y.tofile('y.bin');dy.tofile('dy.bin')
    run=subprocess.run(['./build/validate',str(dtype_id),'7','1'],capture_output=True,text=True,check=True)
    actual=np.fromfile('z.bin',dtype=dtype).astype(np.float32)
    with np.errstate(all='ignore'):
        golden=(dy.astype(np.float32)/np.sqrt(y.astype(np.float32)**2-1)).astype(dtype).astype(np.float32)
    passed=bool(np.array_equal(np.isnan(actual),np.isnan(golden)) and np.array_equal(np.isposinf(actual),np.isposinf(golden)) and np.array_equal(np.isneginf(actual),np.isneginf(golden)) and np.allclose(actual[4:],golden[4:],rtol=.002,atol=.002))
    row=dict(dtype=str(np.dtype(dtype)),boundary=True,actual=[str(x) for x in actual],passed=passed)
    results.append(row); print(json.dumps(row),flush=True)
    assert passed,row
Path('results.json').write_text(json.dumps(results,indent=2))
