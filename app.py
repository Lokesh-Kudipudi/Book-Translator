"""
Book Translator - Gradio App
Translates English documents to Indian languages using IndicTrans2.
Includes side-by-side Original / Translated previews for PDF and TXT.
"""

import gradio as gr
import os
import re
import base64
import html as html_mod
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ── Constants ─────────────────────────────────────────────────────────────────
LANGUAGES = {
    "Hindi":    "hin_Deva",
    "Marathi":  "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Tamil":    "tam_Taml",
    "Telugu":   "tel_Telu",
}

SAMPLE_DIR = Path("Input Samples")
PREVIEW_MAX_PAGES = 4       # max PDF pages to render in preview
PREVIEW_TXT_CHARS = 6000    # max characters to show for TXT preview
PDF_RENDER_ZOOM   = 1.5     # zoom factor for PDF page images

# ── Patch dispatch_model for transformers==4.38.2 + bitsandbytes incompatibility ─
#
# transformers 4.38.2 always calls accelerate's dispatch_model after loading a
# quantized model.  dispatch_model calls model.to(device), which bitsandbytes
# forbids on already-placed quantized weights (ValueError).
#
# Fix: temporarily override .to() on the model class to swallow that ValueError
# so dispatch_model can still move non-quantized modules (embed_positions, layer
# norms, …) to GPU, then restore the original .to() afterwards.

import transformers.modeling_utils as _mu
from transformers.utils.quantization_config import QuantizationMethod as _QM

_orig_dispatch = _mu.dispatch_model

def _bnb_safe_dispatch(model, **kwargs):
    if getattr(model, "quantization_method", None) != _QM.BITS_AND_BYTES:
        return _orig_dispatch(model, **kwargs)

    _cls_to = model.__class__.to

    def _safe_to(self, *args, **kw):
        try:
            return _cls_to(self, *args, **kw)
        except ValueError:
            return self   # already on the correct device — ignore and continue

    model.__class__.to = _safe_to
    try:
        return _orig_dispatch(model, **kwargs)
    finally:
        model.__class__.to = _cls_to  # always restore, even on exception

_mu.dispatch_model = _bnb_safe_dispatch


# ── Move non-quantized tensors to GPU after a bitsandbytes load ───────────────
#
# bitsandbytes only places nn.Linear layers on GPU; other modules
# (positional embeddings, layer norms, etc.) may stay on CPU, causing
# "Expected all tensors to be on the same device" errors at inference time.

def _move_non_bnb_to_gpu(model, device: str = "cuda:0") -> None:
    import bitsandbytes as bnb
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.device.type == "cpu" and not isinstance(
                param, (bnb.nn.Params4bit, bnb.nn.Int8Params)
            ):
                module._parameters[name] = param.to(device)
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                module._buffers[name] = buf.to(device)


# ── Cached model loader ───────────────────────────────────────────────────────
_model_cache: dict = {}

def load_model(quantization=None):
    key = quantization or "full"
    if key in _model_cache:
        return _model_cache[key]

    import torch
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"

    # bitsandbytes quantization requires CUDA — fall back to full precision on CPU
    if quantization in ("4-bit", "8-bit") and DEVICE == "cpu":
        import warnings
        warnings.warn(
            f"{quantization} quantization requires a CUDA GPU; "
            "falling back to full precision on CPU.",
            RuntimeWarning,
        )
        quantization = None
        key = "full"
        if key in _model_cache:
            return _model_cache[key]

    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
        device_map="auto" if qconfig is not None else None,
    )

    if qconfig is None:
        # Full precision: move and cast manually
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model = model.half()
    else:
        # Quantized: bitsandbytes placed Linear layers on GPU; push everything
        # else (positional embeddings, layer norms, …) there too.
        _move_non_bnb_to_gpu(model, device="cuda:0")

    model.eval()

    ip = IndicProcessor(inference=True)
    result = (tokenizer, model, ip, DEVICE)
    _model_cache[key] = result
    return result


# ── Preview helpers ───────────────────────────────────────────────────────────

def _preview_shell(title: str, meta: str, body_html: str) -> str:
    """Wrap preview content in a styled card."""
    return f"""
<div style="
    border:1px solid #e5e7eb; border-radius:10px; overflow:hidden;
    font-family:'Segoe UI',system-ui,sans-serif; background:#fff;
    box-shadow:0 1px 4px rgba(0,0,0,.07);
">
  <div style="
      background:linear-gradient(90deg,#1a1a2e,#0f3460);
      padding:10px 16px; display:flex; align-items:center; gap:10px;
  ">
    <span style="color:#fff;font-weight:600;font-size:.95rem;flex:1">{html_mod.escape(title)}</span>
    <span style="color:#a5b4fc;font-size:.78rem">{html_mod.escape(meta)}</span>
  </div>
  <div style="max-height:520px;overflow-y:auto;padding:14px;background:#fafafa;">
    {body_html}
  </div>
</div>"""


def generate_preview(file_path: str | None, label: str = "Preview") -> str:
    """Return an HTML string previewing a PDF or TXT file."""
    if not file_path:
        return _empty_preview(label)

    path = Path(file_path)
    if not path.exists():
        return _empty_preview(label)

    ext = path.suffix.lower().lstrip(".")

    try:
        if ext == "pdf":
            return _preview_pdf(path, label)
        elif ext == "txt":
            return _preview_txt(path, label)
        else:
            return _empty_preview(label, f"Unsupported type: .{ext}")
    except Exception as e:
        return _empty_preview(label, f"Preview error: {e}")


def _empty_preview(label: str, msg: str = "No file loaded yet.") -> str:
    body = f"""
<div style="
    text-align:center;padding:48px 20px;color:#9ca3af;
    font-size:.9rem;border:2px dashed #e5e7eb;border-radius:8px;
">
  <div style="font-size:2rem;margin-bottom:8px">📄</div>
  {html_mod.escape(msg)}
</div>"""
    return _preview_shell(label, "", body)


def _preview_pdf(path: Path, label: str) -> str:
    import pymupdf as fitz

    doc = fitz.open(str(path))
    total = len(doc)
    shown = min(total, PREVIEW_MAX_PAGES)
    meta  = f"{total} page{'s' if total != 1 else ''}"
    if total > PREVIEW_MAX_PAGES:
        meta += f" (showing first {shown})"

    pages_html = ""
    for i in range(shown):
        page = doc[i]
        mat  = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
        pix  = page.get_pixmap(matrix=mat)
        b64  = base64.b64encode(pix.tobytes("png")).decode()
        pages_html += f"""
<div style="margin-bottom:12px">
  <div style="font-size:.72rem;color:#6b7280;margin-bottom:4px;
              font-weight:500;text-transform:uppercase;letter-spacing:.04em">
    Page {i + 1}
  </div>
  <img src="data:image/png;base64,{b64}"
       style="width:100%;border:1px solid #e5e7eb;border-radius:4px;
              display:block;box-shadow:0 1px 3px rgba(0,0,0,.1)" />
</div>"""
    doc.close()
    return _preview_shell(label, meta, pages_html)


def _preview_txt(path: Path, label: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    total_lines = content.count("\n") + 1
    truncated   = len(content) > PREVIEW_TXT_CHARS
    snippet     = content[:PREVIEW_TXT_CHARS]
    meta        = f"{total_lines} lines"

    escaped = html_mod.escape(snippet)
    note    = (
        f'<div style="color:#6366f1;font-size:.78rem;margin-top:10px;'
        f'font-style:italic">… truncated — showing first {PREVIEW_TXT_CHARS} characters</div>'
        if truncated else ""
    )
    body = f"""
<pre style="
    margin:0; white-space:pre-wrap; word-break:break-word;
    font-family:'Fira Code','Cascadia Code',monospace; font-size:.82rem;
    line-height:1.6; color:#1e293b; background:#f8fafc;
    border:1px solid #e2e8f0; border-radius:6px; padding:12px;
">{escaped}</pre>{note}"""
    return _preview_shell(label, meta, body)


def preview_from_upload(file_path):
    """Called when a file is uploaded."""
    return generate_preview(file_path, label=Path(file_path).name if file_path else "Original")


def preview_from_sample(sample_name):
    """Called when a sample is selected."""
    if not sample_name or sample_name == "— none —":
        return _empty_preview("Original")
    path = SAMPLE_DIR / sample_name
    return generate_preview(str(path), label=sample_name)


# ── Translation helpers ───────────────────────────────────────────────────────
def batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip, device,
                    batch_size=4, progress_cb=None):
    import torch
    translations = []
    total = len(sentences)
    for i in range(0, total, batch_size):
        batch = sentences[i: i + batch_size]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            batch, truncation=True, padding="longest",
            return_tensors="pt", return_attention_mask=True,
        ).to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs, use_cache=True,
                min_length=0, max_length=256,
                num_beams=5, num_return_sequences=1,
            )
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()
        if progress_cb:
            progress_cb(min(i + batch_size, total) / total)
    return translations


def check_text(text: str) -> bool:
    text = text.strip()
    if text in ("", "\t", "•", " "):
        return False
    if all(c == "." for c in text):
        return False
    if re.search(r"\d+[\t\n]", text):
        return False
    try:
        float(text)
        return False
    except ValueError:
        pass
    return not text.isspace()


def translate_pdf(src_path, tgt_path, tgt_lang, model, tokenizer, ip, device,
                  progress_cb=None):
    import pymupdf as fitz
    doc = fitz.open(src_path)
    total_pages = len(doc)
    for page_idx in range(total_pages):
        page = doc[page_idx]
        content_blocks = page.get_text("dict")["blocks"]
        text_blocks, bboxes, font_sizes, colors = [], [], [], []
        for block in content_blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if check_text(span["text"]):
                            text_blocks.append(span["text"])
                            bboxes.append(span["bbox"])
                            font_sizes.append(span["size"])
                            color = fitz.sRGB_to_rgb(span["color"])
                            if color == (255, 255, 255):
                                color = (0, 0, 0)
                            colors.append(color)

        translated = batch_translate(
            text_blocks, "eng_Latn", tgt_lang, model, tokenizer, ip, device,
        )
        for bbox, text, size, color in zip(bboxes, translated, font_sizes, colors):
            css = f"*{{color:rgb{color};font-size:{size}px;}}"
            page.draw_rect(bbox, color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_htmlbox(bbox, text, css=css)

        if progress_cb:
            progress_cb((page_idx + 1) / total_pages)

    doc.save(tgt_path)
    doc.close()


def translate_txt(src_path, tgt_path, tgt_lang, model, tokenizer, ip, device,
                  progress_cb=None):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]
    translated = batch_translate(
        lines, "eng_Latn", tgt_lang, model, tokenizer, ip, device,
        progress_cb=progress_cb,
    )
    with open(tgt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated))


# ── Core translation function ─────────────────────────────────────────────────
def run_translation(
    uploaded_file,
    sample_choice,
    tgt_lang_name,
    quantization_label,
    batch_size,
    progress=gr.Progress(track_tqdm=False),
):
    """Returns (output_filepath, status_md, translated_preview_html)."""
    quant_map = {
        "None (Full Precision)": None,
        "8-bit (less RAM)":      "8-bit",
        "4-bit (least RAM)":     "4-bit",
    }
    tgt_lang_code = LANGUAGES[tgt_lang_name]
    quantization  = quant_map[quantization_label]

    # Resolve source
    if uploaded_file is not None:
        src_orig = Path(uploaded_file)
    elif sample_choice and sample_choice != "— none —":
        src_orig = SAMPLE_DIR / sample_choice
        if not src_orig.exists():
            raise gr.Error(f"Sample file not found: {src_orig}")
    else:
        raise gr.Error("Please upload a file or choose a sample.")

    ext = src_orig.suffix.lower().lstrip(".")
    if ext not in ("pdf", "txt"):
        raise gr.Error(f"Unsupported file type: .{ext}")

    tmp_dir = tempfile.mkdtemp()
    try:
        src = Path(tmp_dir) / src_orig.name
        shutil.copy(src_orig, src)

        out_name = src.stem + f"_{tgt_lang_name}" + src.suffix
        out_path = Path(tmp_dir) / out_name

        progress(0, desc="Loading model…")
        tokenizer, model, ip, device = load_model(quantization)

        def update(val):
            progress(val, desc=f"Translating… {int(val * 100)}%")

        progress(0.01, desc="Starting translation…")
        if ext == "pdf":
            translate_pdf(str(src), str(out_path), tgt_lang_code,
                          model, tokenizer, ip, device, progress_cb=update)
        else:
            translate_txt(str(src), str(out_path), tgt_lang_code,
                          model, tokenizer, ip, device, progress_cb=update)

        progress(1.0, desc="Done!")

        # Save to a stable directory outside tmp_dir
        stable_dir = tempfile.mkdtemp()
        stable_out = Path(stable_dir) / out_name
        shutil.copy(out_path, stable_out)

        translated_preview = generate_preview(
            str(stable_out),
            label=f"{out_name} ({tgt_lang_name})",
        )
        status = f"✅ Translation complete → **{out_name}**"
        return str(stable_out), status, translated_preview

    except Exception as e:
        raise gr.Error(str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Sample file list ──────────────────────────────────────────────────────────
def get_sample_files():
    if not SAMPLE_DIR.exists():
        return ["— none —"]
    files = [f.name for f in SAMPLE_DIR.iterdir()
             if f.suffix.lower() in (".pdf", ".txt")]
    return ["— none —"] + files if files else ["— none —"]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
custom_css = """
body, .gradio-container { font-family: 'Segoe UI', system-ui, sans-serif; }

#app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 12px; padding: 28px 32px 22px; margin-bottom: 8px;
}
#app-header h1 { color:#fff; font-size:2rem; font-weight:700; margin:0 0 6px; }
#app-header p  { color:#a5b4fc; font-size:.95rem; margin:0; }

#lang-badges { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 16px; }
#lang-badges span {
    background:#e0e7ff; color:#3730a3;
    padding:3px 12px; border-radius:20px; font-size:.82rem; font-weight:500;
}

.section-label {
    font-size:1rem; font-weight:600; color:#374151;
    border-left:4px solid #6366f1; padding-left:10px; margin-bottom:6px;
}

#translate-btn {
    background:#6366f1 !important; border-radius:8px !important;
    font-weight:600 !important; margin-top:8px !important;
}
#translate-btn:hover { background:#4f46e5 !important; }

/* Make preview HTML boxes scroll cleanly */
.preview-col > div { min-height: 200px; }
"""

with gr.Blocks(css=custom_css, title="Book Translator") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-header">
      <h1>📖 Book Translator</h1>
      <p>Translate English documents to Indian languages using the IndicTrans2 model.</p>
    </div>
    <div id="lang-badges">
      <span>Hindi</span><span>Marathi</span><span>Gujarati</span>
      <span>Tamil</span><span>Telugu</span>
    </div>
    """)

    # ── Controls row ──────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML('<p class="section-label">Choose Input</p>')
            with gr.Tabs():
                with gr.TabItem("📤 Upload a File"):
                    uploaded_file = gr.File(
                        label="Upload a PDF or TXT file",
                        file_types=[".pdf", ".txt"],
                        type="filepath",
                    )
                with gr.TabItem("📂 Use Sample File"):
                    sample_choice = gr.Dropdown(
                        label="Pick a sample file",
                        choices=get_sample_files(),
                        value="— none —",
                    )
                    gr.HTML("<small style='color:#6b7280'>Add .pdf or .txt files to "
                            "<code>Input Samples/</code> to populate this list.</small>")

        with gr.Column(scale=2):
            gr.HTML('<p class="section-label">Settings</p>')
            tgt_lang = gr.Dropdown(
                label="Target Language",
                choices=list(LANGUAGES.keys()),
                value="Hindi",
            )
            quantization = gr.Dropdown(
                label="Quantization",
                choices=["None (Full Precision)", "8-bit (less RAM)", "4-bit (least RAM)"],
                value="None (Full Precision)",
                info="Reduce GPU/CPU memory at the cost of some accuracy.",
            )
            batch_size = gr.Slider(
                label="Batch Size", minimum=1, maximum=16, value=4, step=1,
                info="Larger = faster but uses more memory.",
            )
            translate_btn = gr.Button(
                "Translate →", variant="primary", elem_id="translate-btn",
            )

    # ── Status + download ─────────────────────────────────────────────────────
    with gr.Row():
        status_box  = gr.Markdown(
            value="Upload or select a file, then click **Translate →**.",
            label="Status",
        )
        output_file = gr.File(
            label="⬇ Download Translated File",
            interactive=False,
        )

    # ── Side-by-side preview ──────────────────────────────────────────────────
    gr.HTML('<p class="section-label" style="margin-top:20px">Preview</p>')
    gr.HTML("<small style='color:#6b7280;display:block;margin-bottom:10px'>"
            "PDF: first 4 pages rendered as images &nbsp;·&nbsp; "
            "TXT: first 6 000 characters shown</small>")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["preview-col"]):
            orig_preview = gr.HTML(
                value=_empty_preview("Original"),
                label="Original",
            )
        with gr.Column(elem_classes=["preview-col"]):
            trans_preview = gr.HTML(
                value=_empty_preview("Translated"),
                label="Translated",
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <hr style="margin-top:28px;border-color:#e5e7eb"/>
    <p style="text-align:center;color:#9ca3af;font-size:.82rem">
      Book Translator · <a href="https://github.com/AI4Bharat/IndicTrans2"
      target="_blank" style="color:#6366f1">IndicTrans2</a> by AI4Bharat ·
      MIT License · Supports PDF &amp; TXT
    </p>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────

    # Show original preview on upload
    uploaded_file.change(
        fn=preview_from_upload,
        inputs=[uploaded_file],
        outputs=[orig_preview],
    )

    # Show original preview on sample selection
    sample_choice.change(
        fn=preview_from_sample,
        inputs=[sample_choice],
        outputs=[orig_preview],
    )

    # Clear translated preview when a new file is chosen
    def clear_translated():
        return _empty_preview("Translated")

    uploaded_file.change(fn=clear_translated, inputs=[], outputs=[trans_preview])
    sample_choice.change(fn=clear_translated, inputs=[], outputs=[trans_preview])

    # Run translation → update download, status, translated preview
    translate_btn.click(
        fn=run_translation,
        inputs=[uploaded_file, sample_choice, tgt_lang, quantization, batch_size],
        outputs=[output_file, status_box, trans_preview],
    )


if __name__ == "__main__":
    demo.launch(share=True)