"""Run every experiment; results land in ../../results."""
import pathlib
import runpy
import time

HERE = pathlib.Path(__file__).parent
for name in ("exp1_invariance", "exp2_ray", "exp3_slices"):
    t = time.time()
    print(f"== {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
    print(f"== {name} done in {time.time() - t:.0f}s")
