"""
diagnose_path.py — 诊断 src 同名包冲突
用法: python tools/diagnose_path.py
"""
import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

print("=" * 70)
print("当前 sys.path 顺序（前面优先）:")
print("=" * 70)
for i, p in enumerate(sys.path):
    mark = ""
    if "boundary_prefill" in str(p):
        mark = "   <=== 含 boundary_prefill"
    elif str(ROOT) == str(p):
        mark = "   <=== 项目根"
    print(f"  [{i}] {p}{mark}")

print()
print("=" * 70)
print("查找所有名为 'src' 的目录:")
print("=" * 70)
src_dirs = list(ROOT.glob("**/src"))
for d in src_dirs:
    print(f"  {d}")

print()
print("=" * 70)
print("模拟: import src.utils.io 会加载哪个文件?")
print("=" * 70)

# 保存原始 sys.path
original = list(sys.path)

# 确保项目根在 path 中（模拟 video2json.py 的行为）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # 先卸载可能已经缓存的 src
    for mod_name in list(sys.modules.keys()):
        if mod_name == "src" or mod_name.startswith("src."):
            del sys.modules[mod_name]

    import src.utils.io as io_mod
    print(f"  ✅ 实际加载: {io_mod.__file__}")

    # 检查是否有 load_yaml_with_base
    has_func = hasattr(io_mod, "load_yaml_with_base")
    print(f"  包含 load_yaml_with_base: {has_func}")

    if not has_func:
        print()
        print("  ❌ 这是错误的 src！来自 boundary_prefill_B_final")
        print("     因为 sys.path 中 boundary 相关路径排在前面")
except ImportError as e:
    print(f"  ❌ 导入失败: {e}")
finally:
    # 恢复 sys.path
    sys.path = original

print()
print("=" * 70)
print("诊断建议:")
print("=" * 70)
boundary_paths = [p for p in sys.path if "boundary_prefill" in str(p)]
if boundary_paths:
    print("  ⚠️ 发现 boundary_prefill 在 sys.path 中:")
    for p in boundary_paths:
        print(f"    - {p}")
    print()
    print("  → 必须在 run_all() 调用前移除这些路径")
else:
    print("  ✅ sys.path 中无 boundary_prefill（如果是在干净状态下运行）")

print()
print("  location_channel 的 src 应该在:")
loc_src = ROOT / "location_channel" / "src"
print(f"    {loc_src}")
print(f"    存在: {loc_src.exists()}")

# 检查 location_channel 的 io.py 是否有 load_yaml_with_base
io_file = loc_src / "utils" / "io.py"
print()
print(f"  location_channel/src/utils/io.py 存在: {io_file.exists()}")
if io_file.exists():
    content = io_file.read_text(encoding="utf-8", errors="ignore")
    if "load_yaml_with_base" in content:
        print(f"  ✅ 包含 load_yaml_with_base 定义")
    else:
        print(f"  ❌ 不包含 load_yaml_with_base（需要检查！）")