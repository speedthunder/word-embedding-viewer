"""詞向量運算核心：文字 → 向量 → 3D 座標 / 分群 / 最近鄰。
來源：
  semantic  多語言語意模型（paraphrase-multilingual-MiniLM-L12-v2，中英文皆可）
  gpt2-wte  GPT-2 輸入嵌入層（詞嵌入矩陣 wte；多 token 的詞取平均；中文會被拆成位元組 token，效果差）
"""
import base64
import re
import threading

import numpy as np

SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GPT2_MODEL = "openai-community/gpt2"
MAX_ITEMS = 3000
MAX_CHARS = 300

_lock = threading.Lock()
_cache = {}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(source):
    if source in _cache:
        return _cache[source]
    from transformers import AutoModel, AutoTokenizer
    name = SEMANTIC_MODEL if source == "semantic" else GPT2_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(_device()).eval()
    _cache[source] = (tok, model)
    return _cache[source]


def parse_items(text):
    """每行一筆。分組只認兩種寫法：「文字<Tab>分組」或「文字,分組」（半形逗號、後面不能接空白）。
    全形逗號「，」與英文句子裡的「, 」都不是分組分隔，所以一般句子不會被誤拆。"""
    items, seen = [], set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        group = ""
        m = re.match(r"^(.*?)(?:\t|,(?=\S))\s*([^\t,，]{1,30})$", line)
        if m and len(m.group(1).strip()) > 0:
            line, group = m.group(1).strip(), m.group(2).strip()
        line = line[:MAX_CHARS]
        key = (line, group)
        if key in seen:
            continue
        seen.add(key)
        items.append({"t": line, "g": group})
    return items


def embed(texts, source="semantic", batch=64):
    import torch
    tok, model = _load(source)
    dev = _device()
    out = []
    if source == "gpt2-wte":
        wte = model.wte.weight.detach().cpu().numpy()
        for t in texts:
            ids = tok.encode(" " + t)  # 前置空白，對應 GPT-2 的詞首 token
            out.append(wte[ids].mean(axis=0))
        return np.asarray(out, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def _norm3(xyz):
    xyz = xyz - xyz.mean(axis=0)
    s = np.abs(xyz).max()
    return (xyz / s if s > 0 else xyz).astype(np.float32)


def reduce3(vecs, method):
    from sklearn.decomposition import PCA
    n = len(vecs)
    if n < 4:
        pad = np.zeros((n, 3), dtype=np.float32)
        pad[:, 0] = np.arange(n)
        return _norm3(pad)
    if method == "pca":
        return _norm3(PCA(n_components=3, random_state=0).fit_transform(vecs))
    if method == "tsne":
        from sklearn.manifold import TSNE
        pre = PCA(n_components=min(50, vecs.shape[1], n - 1), random_state=0).fit_transform(vecs)
        perp = max(2, min(30, (n - 1) // 3))
        return _norm3(TSNE(n_components=3, method="exact" if n <= 1500 else "barnes_hut",
                           perplexity=perp, init="pca", random_state=0, metric="cosine"
                           ).fit_transform(pre)) if n <= 1500 else None
    raise ValueError(method)


def cluster(vecs, k):
    from sklearn.cluster import KMeans
    n = len(vecs)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs)


def neighbors(vecs, top=10):
    v = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    sim = v @ v.T
    np.fill_diagonal(sim, -2)
    top = min(top, len(vecs) - 1)
    if top < 1:
        return [[] for _ in range(len(vecs))]
    idx = np.argsort(-sim, axis=1)[:, :top]
    return [[[int(j), round(float(sim[i, j]), 4)] for j in idx[i]] for i in range(len(vecs))]


def build(text, source="semantic", k=0):
    """回傳可直接給前端的 dict。k=0 表示自動估計分群數。"""
    items = parse_items(text)
    if len(items) < 2:
        raise ValueError("至少需要 2 筆資料（每行一筆）")
    truncated = len(items) > MAX_ITEMS
    items = items[:MAX_ITEMS]
    with _lock:
        vecs = embed([it["t"] for it in items], source)
    n = len(items)
    if k <= 0:
        k = int(np.clip(round(np.sqrt(n / 2)), 2, 12))
    groups = sorted({it["g"] for it in items if it["g"]})
    labels = cluster(vecs, k)
    pca = reduce3(vecs, "pca")
    tsne = reduce3(vecs, "tsne")
    nn = neighbors(vecs)
    res = {
        "meta": {"source": source, "model": SEMANTIC_MODEL if source == "semantic" else GPT2_MODEL,
                 "n": n, "dim": int(vecs.shape[1]), "k": int(k), "groups": groups, "truncated": truncated,
                 "tsne": tsne is not None},
        "items": [{"t": it["t"], "g": it["g"], "c": int(labels[i]),
                   "p": [round(float(x), 4) for x in pca[i]],
                   **({"s": [round(float(x), 4) for x in tsne[i]]} if tsne is not None else {}),
                   "nn": nn[i]} for i, it in enumerate(items)],
        "vec": base64.b64encode(vecs.astype("<f4").tobytes()).decode("ascii"),
    }
    return res, vecs


def analogy(vecs, a, b, c, top=8):
    """A − B + C 的最近鄰（排除 a, b, c 本身）。"""
    v = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    q = v[a] - v[b] + v[c]
    q = q / max(np.linalg.norm(q), 1e-9)
    sim = v @ q
    for i in (a, b, c):
        sim[i] = -2
    idx = np.argsort(-sim)[:top]
    return [[int(i), round(float(sim[i]), 4)] for i in idx]
