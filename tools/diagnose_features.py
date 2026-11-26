import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_prefill_B_final.src.features.builder import FeatureBuilder, load_pose_json, build_hand_features
import inspect

print("=== load_pose_json 源码 ===")
print(inspect.getsource(load_pose_json))

print("\n=== build_hand_features 签名 ===")
print(inspect.signature(build_hand_features))

print("\n=== build_hand_features 源码前40行 ===")
src = inspect.getsource(build_hand_features)
for i, line in enumerate(src.split('\n')[:40]):
    print(f"  {line}")

# 测试：创建一个假的关键点 JSON，看能不能跑通
import json
import tempfile
import numpy as np

# 模拟 MediaPipe 输出格式：(71, 21, 3)
T = 71
fake_xyz = np.random.randn(T, 21, 3).astype(np.float32)

# 保存成 JSON（猜测格式）
fake_data = {
    "pose": fake_xyz.tolist(),  # 或者可能是 {"landmarks": [...]}
}
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(fake_data, f)
    tmp_path = f.name

print(f"\n=== 测试 load_pose_json ===")
try:
    loaded = load_pose_json(tmp_path)
    print(f"  成功！类型: {type(loaded)}")
    if isinstance(loaded, dict):
        for k in list(loaded.keys())[:5]:
            v = loaded[k]
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}")
            else:
                print(f"    {k}: {type(v)}")
except Exception as e:
    print(f"  失败: {e}")
    import traceback
    traceback.print_exc()

import os
os.unlink(tmp_path)