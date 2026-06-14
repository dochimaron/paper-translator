#!/usr/bin/env python3
"""
paper_translator.py
MinerU의 content_list.json을 입력받아, 전문을 번역한 좌우 대조 HTML을 생성한다.
백엔드: Google Gemini (무료 티어 사용 가능)
 - text 블록   : Gemini로 한국어 번역 (블록 단위 → 좌우 정렬 유지)
 - image/table : MinerU가 잘라낸 img_path를 그대로 삽입 (왜곡 없음, 번역 X)
 - equation    : LaTeX를 KaTeX로 렌더 + 중2 수준 설명 박스 자동 생성
 - 캡션/각주   : 함께 번역

사용:
  pip install google-genai mineru
  export GEMINI_API_KEY=...                  # aistudio.google.com 에서 발급
  mineru -p paper.pdf -o ./out
  python paper_translator.py ./out/paper/paper_content_list.json -o paper_ko.html
"""

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"     # 한도 더 여유로우려면 "gemini-2.5-flash-lite"
BATCH_CHARS = 6000                     # 번역 1회 호출당 묶는 글자 수 한도
MAX_RETRIES = 5                        # 무료 티어 429(분당 한도)는 백오프로 흡수
TRANSLATABLE = {"text", "list"}        # 번역 대상 타입
SKIP_TYPES = {"header", "footer", "page_number", "page_footnote", "aside_text"}
# 캡션 필드 순서 — translate_blocks와 build_html이 같은 순서를 써야 key가 맞는다
CAP_FIELDS = ("image_caption", "chart_caption", "table_caption",
              "image_footnote", "table_footnote")

TRANSLATE_SYSTEM = """당신은 영어 학술 논문을 한국어로 번역하는 전문 번역가다. 규칙:
- 학술적이고 자연스러운 한국어 문어체로 번역한다.
- (Author, 2023), (Kim et al., 2024) 같은 인용 표기는 원문 그대로 둔다.
- 수식 기호, 변수명, 통계 용어는 학계 관례를 따른다 (예: Difference-in-Differences → 이중차분).
- 입력은 {"블록번호": "원문"} 형태의 JSON이다.
- 출력은 {"블록번호": "번역문"} 형태의 JSON만 반환한다. 설명·코드펜스 없이 순수 JSON만."""

EQUATION_SYSTEM = """당신은 어려운 수식을 중학교 2학년도 이해할 수 있게 풀어주는 선생님이다. 규칙:
- 수식에 나오는 각 기호가 무엇을 뜻하는지 쉬운 말로 하나씩 설명한다.
- Σ(시그마)는 '다 더한다', 아래첨자는 '이름표'처럼 풀어서 설명한다.
- 마지막에 "한마디로:" 로 시작하는 한 문장 요약을 붙인다.
- 전문 용어를 최소화하고, 2~4문장 정도로 간결하게. 한국어로만 답한다.
- 설명 텍스트만 반환한다 (머리말·코드펜스 없이)."""


# ──────────────────────────────────────────────────────────────────────────
# LLM 호출 (Gemini, 재시도 포함)
# ──────────────────────────────────────────────────────────────────────────
def call_llm(client, model, system, user, max_tokens=8000):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  ! API 오류({e!r}), {wait}s 후 재시도 {attempt+1}/{MAX_RETRIES}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("API 호출이 반복 실패했습니다.")


def parse_json(text):
    """코드펜스가 섞여 와도 JSON만 추출."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


# ──────────────────────────────────────────────────────────────────────────
# 캐시 (재실행 시 같은 블록 재번역 방지 → 호출 한도/비용 절감)
# ──────────────────────────────────────────────────────────────────────────
class Cache:
    def __init__(self, path, enabled=True):
        self.path = path
        self.enabled = enabled
        self.data = {}
        if enabled and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(*parts):
        return hashlib.sha256("\u241F".join(parts).encode("utf-8")).hexdigest()

    def get(self, k):
        return self.data.get(k) if self.enabled else None

    def set(self, k, v):
        if self.enabled:
            self.data[k] = v

    def flush(self):
        if self.enabled:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# 블록 유틸
# ──────────────────────────────────────────────────────────────────────────
def block_text(b):
    if b.get("list_items"):                 # VLM 백엔드 list
        return "\n".join(b["list_items"])
    return (b.get("text") or "").strip()


def clean_latex(raw):
    s = (raw or "").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return re.sub(r"\s*\n\s*", " ", s)


def captions(b, *fields):
    out = []
    for f in fields:
        v = b.get(f)
        if isinstance(v, list):
            out.extend(x for x in v if x)
        elif v:
            out.append(v)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 번역: text/list 본문 + 캡션/각주 일괄
# ──────────────────────────────────────────────────────────────────────────
def translate_blocks(blocks, client, model, cache, do_caption=True):
    """{원본 인덱스 또는 'cap:i:n': 번역문} 반환. 블록 단위 유지 → 좌우 정렬 보존."""
    result = {}
    pending = []   # (result_key, text)

    def add(result_key, txt):
        txt = (txt or "").strip()
        if not txt:
            return
        ck = Cache.key("tr", txt)
        cached = cache.get(ck)
        if cached is not None:
            result[result_key] = cached
        else:
            pending.append((result_key, txt))

    for i, b in enumerate(blocks):
        t = b.get("type")
        if t in TRANSLATABLE:
            add(i, block_text(b))                        # 본문 → key=i
        elif do_caption and t in ("image", "chart", "table"):
            for n, c in enumerate(captions(b, *CAP_FIELDS)):
                add(f"cap:{i}:{n}", c)                    # 캡션/각주 → key='cap:i:n'

    # 글자 수 기준으로 배치 분할
    batch, size, batches = [], 0, []
    for key, txt in pending:
        if size + len(txt) > BATCH_CHARS and batch:
            batches.append(batch)
            batch, size = [], 0
        batch.append((key, txt))
        size += len(txt)
    if batch:
        batches.append(batch)

    for n, bt in enumerate(batches, 1):
        print(f"  번역 배치 {n}/{len(batches)} ({len(bt)}블록)…")
        payload = {str(key): txt for key, txt in bt}
        out = call_llm(client, model, TRANSLATE_SYSTEM,
                       json.dumps(payload, ensure_ascii=False))
        translated = parse_json(out)
        for key, txt in bt:
            ko = translated.get(str(key), "(번역 누락)")
            result[key] = ko
            cache.set(Cache.key("tr", txt), ko)
    return result


# ──────────────────────────────────────────────────────────────────────────
# 수식 설명 생성
# ──────────────────────────────────────────────────────────────────────────
def explain_equations(blocks, client, model, cache):
    out = {}
    eqs = [(i, b) for i, b in enumerate(blocks) if b.get("type") == "equation"]
    for n, (i, b) in enumerate(eqs, 1):
        latex = clean_latex(b.get("text", ""))
        if not latex:
            continue
        ctx = ""
        for j in range(i - 1, -1, -1):
            if blocks[j].get("type") in TRANSLATABLE:
                ctx = block_text(blocks[j])[:600]
                break
        ck = Cache.key("eq", latex, ctx)
        cached = cache.get(ck)
        if cached is not None:
            out[i] = cached
            continue
        print(f"  수식 설명 {n}/{len(eqs)}…")
        user = f"수식(LaTeX): {latex}\n\n앞 문맥: {ctx or '(없음)'}"
        exp = call_llm(client, model, EQUATION_SYSTEM, user, max_tokens=1000)
        out[i] = exp
        cache.set(ck, exp)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 이미지 처리
# ──────────────────────────────────────────────────────────────────────────
def img_src(img_path, base_dir, embed):
    if not img_path:
        return None
    p = (base_dir / img_path)
    if embed and p.exists():
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    return img_path  # 상대 경로


# ──────────────────────────────────────────────────────────────────────────
# HTML 생성
# ──────────────────────────────────────────────────────────────────────────
CSS = """
.doc{max-width:1120px;margin:0 auto;padding:8px 16px 48px;
 font-family:-apple-system,"Segoe UI","Noto Sans KR",system-ui,sans-serif;color:#1f2937}
.colhead{display:grid;grid-template-columns:1fr 1fr;gap:28px;position:sticky;top:0;
 background:#fff;padding:8px 0;border-bottom:1px solid #e5e7eb;font-size:12px;
 font-weight:600;color:#9ca3af;z-index:5}
.row{display:grid;grid-template-columns:1fr 1fr;gap:28px;padding:12px 0;
 border-bottom:1px solid #f1f5f9;align-items:start}
.orig{color:#6b7280;font-size:13.5px;line-height:1.75}
.trans{color:#111827;font-size:14.5px;line-height:1.85}
h2.sec{grid-column:1/-1;font-size:17px;margin:22px 0 4px;padding-bottom:4px;
 border-bottom:1px solid #e5e7eb;color:#2563eb}
h2.sec .en{color:#9ca3af;font-weight:500;font-size:12.5px;display:block}
.eq-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
 padding:16px 18px;margin:14px 0 0;overflow-x:auto}
.eq-explain{margin:10px 0 18px;background:#eff6ff;border-left:4px solid #3b82f6;
 padding:12px 16px;border-radius:6px;font-size:13.5px;line-height:1.8}
.eq-explain .tag{font-weight:700;color:#1d4ed8}
.fig{margin:8px 0 0;text-align:center}
.fig img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}
.cap{font-size:12.5px;color:#4b5563;margin:6px 2px 14px;line-height:1.7}
"""

HTML_HEAD = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>{css}</style></head><body><div class="doc">
<div class="colhead"><div>원문 (English)</div><div>번역 (한국어)</div></div>
"""

HTML_TAIL = """</div>
<script>
document.addEventListener("DOMContentLoaded",function(){
 renderMathInElement(document.body,{delimiters:[
  {left:"$$",right:"$$",display:true},{left:"$",right:"$",display:false}],
  throwOnError:false});});
</script></body></html>"""


def esc(s):
    return html.escape(s or "")


def build_html(blocks, trans, expl, base_dir, embed, title, do_caption):
    parts = [HTML_HEAD.format(title=esc(title), css=CSS)]
    for i, b in enumerate(blocks):
        t = b.get("type")
        if t in SKIP_TYPES:
            continue

        if t in TRANSLATABLE:
            orig = block_text(b)
            if not orig:
                continue
            ko = trans.get(i, "")
            if b.get("text_level"):  # 제목 → 전체 폭 섹션 헤더
                parts.append(f'<h2 class="sec">{esc(ko)}<span class="en">{esc(orig)}</span></h2>')
            else:
                parts.append(
                    f'<div class="row"><div class="orig">{esc(orig)}</div>'
                    f'<div class="trans">{esc(ko)}</div></div>'
                )

        elif t == "equation":
            latex = clean_latex(b.get("text", ""))
            parts.append(f'<div class="eq-box">$$ {esc(latex)} $$</div>')
            if i in expl:
                parts.append(
                    f'<div class="eq-explain"><span class="tag">💡 쉬운 설명</span><br>'
                    f'{esc(expl[i])}</div>'
                )

        elif t in ("image", "chart", "table"):
            src = img_src(b.get("img_path"), base_dir, embed)
            if src:
                parts.append(f'<div class="fig"><img src="{esc(src)}" alt="{t}"></div>')
            else:
                parts.append(f'<div class="cap">[{t} 이미지 없음: img_path 비어있음]</div>')
            if do_caption:
                for n, c in enumerate(captions(b, *CAP_FIELDS)):
                    ko = trans.get(f"cap:{i}:{n}", c)   # 번역 있으면 번역, 없으면 원문
                    parts.append(f'<div class="cap">{esc(ko)}</div>')
    parts.append(HTML_TAIL)
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MinerU content_list.json → 좌우 대조 번역 HTML (Gemini)")
    ap.add_argument("content_list", help="MinerU의 *_content_list.json 경로")
    ap.add_argument("-o", "--out", default="output_ko.html", help="출력 HTML 경로")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="사용할 Gemini 모델")
    ap.add_argument("-t", "--title", default="논문 번역 (좌우 대조)", help="문서 제목")
    ap.add_argument("--no-embed", action="store_true",
                    help="이미지를 base64로 넣지 않고 상대경로로 참조")
    ap.add_argument("--no-caption", action="store_true", help="캡션 번역/표시 생략")
    ap.add_argument("--no-cache", action="store_true", help="번역 캐시 비활성화")
    args = ap.parse_args()

    cl_path = Path(args.content_list)
    blocks = json.loads(cl_path.read_text(encoding="utf-8"))
    base_dir = cl_path.parent
    cache = Cache(cl_path.with_suffix(".trcache.json"), enabled=not args.no_cache)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"블록 {len(blocks)}개 로드. 모델: {args.model}")
    print("1) 텍스트 + 캡션 번역…")
    trans = translate_blocks(blocks, client, args.model, cache,
                             do_caption=not args.no_caption)
    print("2) 수식 설명 생성…")
    expl = explain_equations(blocks, client, args.model, cache)
    cache.flush()

    print("3) HTML 생성…")
    out_html = build_html(blocks, trans, expl, base_dir,
                          embed=not args.no_embed, title=args.title,
                          do_caption=not args.no_caption)
    Path(args.out).write_text(out_html, encoding="utf-8")
    print(f"완료 → {args.out}  (번역 {len(trans)}블록, 수식 {len(expl)}개)")


if __name__ == "__main__":
    main()
