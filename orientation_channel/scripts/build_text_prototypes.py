import argparse
from pathlib import Path
import torch


def clip_model_to_hf_id(clip_model: str) -> str:
    m = str(clip_model).strip()
    mapping = {
        'ViT-L-14': 'openai/clip-vit-large-patch14',
        'ViT-B-32': 'openai/clip-vit-base-patch32',
        'ViT-B-16': 'openai/clip-vit-base-patch16',
    }
    return mapping.get(m, 'openai/clip-vit-large-patch14')


def build_with_open_clip(labels, prompts, args):
    import open_clip  # type: ignore

    model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.pretrained)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    embs = []
    with torch.no_grad():
        for lab in labels:
            texts = prompts.get(lab, [lab])
            tokens = tokenizer(texts).to(args.device)
            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            proto = text_feat.mean(dim=0)
            proto = proto / proto.norm()
            embs.append(proto.cpu())
    return torch.stack(embs, dim=0)


def build_with_transformers(labels, prompts, args):
    from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore

    model_id = clip_model_to_hf_id(args.clip_model)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPTextModel.from_pretrained(model_id).to(args.device).eval()

    embs = []
    with torch.no_grad():
        for lab in labels:
            texts = prompts.get(lab, [lab])
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to(args.device)
            out = model(**tokenized)
            # CLIP text embedding at EOS position through pooler_output.
            text_feat = out.pooler_output
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            proto = text_feat.mean(dim=0)
            proto = proto / proto.norm()
            embs.append(proto.cpu())
    return torch.stack(embs, dim=0)


def load_labels(path: str):
    labels = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        labels.append(line)
    return labels


def load_prompts(path: str):
    prompts = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '\t' in line:
            lab, text = line.split('\t', 1)
        else:
            lab, text = line.split(',', 1)
        prompts.setdefault(lab, []).append(text)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_file', required=True)
    ap.add_argument('--prompt_file', required=True)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--clip_model', default='ViT-L-14')
    ap.add_argument('--pretrained', default='openai')
    ap.add_argument('--backend', choices=['auto', 'open_clip', 'transformers'], default='auto')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    labels = load_labels(args.labels_file)
    prompts = load_prompts(args.prompt_file)

    backend_used = args.backend
    if backend_used == 'auto':
        try:
            import open_clip  # noqa: F401
            backend_used = 'open_clip'
        except Exception:
            backend_used = 'transformers'

    if backend_used == 'open_clip':
        E = build_with_open_clip(labels, prompts, args)
    else:
        E = build_with_transformers(labels, prompts, args)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(E, args.out_path)
    print('saved', args.out_path, 'shape', tuple(E.shape), 'backend', backend_used)


if __name__ == '__main__':
    main()
