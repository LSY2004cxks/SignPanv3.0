import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_prefill_B_final.src.features.builder import FeatureBuilder
import inspect

print("=== FeatureBuilder.__init__ 签名 ===")
print(inspect.signature(FeatureBuilder.__init__))

print("\n=== FeatureBuilder 所有公开方法 ===")
for name, method in inspect.getmembers(FeatureBuilder, predicate=inspect.isfunction):
    if not name.startswith('_'):
        print(f"  {name}: {inspect.signature(method)}")

print("\n=== FeatureBuilder.build() 源码前30行 ===")
try:
    src = inspect.getsource(FeatureBuilder.build)
    for i, line in enumerate(src.split('\n')[:30]):
        print(f"  {line}")
except Exception as e:
    print(f"  (无法获取: {e})")

print("\n=== 尝试实例化 ===")
try:
    builder = FeatureBuilder()
    print(f"✅ 默认实例化成功")
except Exception as e:
    print(f"❌ 默认实例化失败: {e}")
    # 尝试用 model_cfg 里的参数
    try:
        builder = FeatureBuilder(pose_dim=23, flow_dim=6, fusion_mode='early',
                                  fusion_dim=64, gate_hidden=64)
        print(f"✅ 带参实例化成功")
    except Exception as e2:
        print(f"❌ 带参也失败: {e2}")