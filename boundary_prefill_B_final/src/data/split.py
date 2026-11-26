from typing import Dict, List, Optional

from ..utils.io import load_json


def load_signer_split(path: str) -> Dict:
    return load_json(path)


def get_split_signers(
    split_obj: Dict,
    split_name: str,
    override_signers: Optional[List[str]] = None,
) -> List[str]:
    if override_signers:
        return list(override_signers)
    if split_name not in split_obj:
        raise KeyError(f"split {split_name} not in split file")
    val = split_obj[split_name]
    if not isinstance(val, list):
        raise TypeError(f"split {split_name} must be list")
    return [str(x) for x in val]
