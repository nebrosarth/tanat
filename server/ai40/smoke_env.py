"""Transport/simulation smoke test; requires only NumPy, not PyTorch."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tanat_ai40.env import AssaultEnvProcess, HeroAction  # noqa: E402

executable = Path(sys.argv[1] if len(sys.argv) > 1 else "../assaultenv.exe").resolve()
with AssaultEnvProcess(executable) as env:
    result = env.reset(seed=123, max_steps=50)
    for _ in range(45):
        result = env.step([HeroAction() for _ in range(10)])
    print(f"ok schema={result.step}:{result.elapsed:.1f}s entities={int(result.entity_mask.sum())} reward={result.rewards.sum():.3f}")
