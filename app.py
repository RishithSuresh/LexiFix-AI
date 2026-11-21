"""Modern desktop and web application for LexiFix AI.

Provides a next-generation interactive studio for spelling detection, explainable
candidate ranking, Transformer model comparisons, and benchmark dataset exploration.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

from spelling_correction import (
    DEFAULT_CORPUS,
    LLMCorrector,
    NoisyChannelSpellCorrector,
    _damerau_levenshtein_distance,
    _edit_distance,
    _soundex,
    character_error_rate,
    correction_metrics,
    sentence_exact_match,
    word_accuracy,
    word_error_rate,
)
from run_comparison import MODEL_PRESETS, load_cases, run as run_cli_cases

MODEL_OPTIONS = (
    "vennify/t5-base-grammar-correction",
    "vennify/t5-small-grammar-correction",
    "google/flan-t5-small",
    "oliverguhr/spelling-correction-english-base",
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Thread-safe LLM model cache
_LLM_CACHE: dict[str, LLMCorrector] = {}
_LLM_LOCK = threading.Lock()
_BASELINE = NoisyChannelSpellCorrector()


def get_llm(model_name: str) -> LLMCorrector:
    """Retrieve or lazily initialize an LLM corrector."""
    with _LLM_LOCK:
        if model_name not in _LLM_CACHE:
            prefix = MODEL_PRESETS.get(model_name, "grammar:")
            _LLM_CACHE[model_name] = LLMCorrector(model_name, task_prefix=prefix)
        return _LLM_CACHE[model_name]


def find_free_port(preferred: int = 8765) -> int:
    """Find an available TCP port starting from preferred."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


class LexiFixRequestHandler(SimpleHTTPRequestHandler):
    """Multi-threaded REST API and static asset handler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress verbose standard HTTP request logs for cleaner CLI
        pass

    def do_GET(self) -> None:
        """Route GET requests to API endpoints or static web assets."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/info":
            self.send_json(self._get_sys_info())
        elif path == "/api/csv/list":
            self.send_json(self._get_csv_list())
        elif path == "/api/csv/read":
            params = urllib.parse.parse_qs(parsed.query)
            filename = params.get("file", [""])[0]
            self.send_json(self._read_csv_file(filename))
        elif path == "/api/vocabulary":
            self.send_json(self._get_vocabulary())
        else:
            # Static file resolution
            if path in ("", "/"):
                self.path = "/index.html"
            return super().do_GET()

    def do_POST(self) -> None:
        """Route POST requests to API action endpoints."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()

        if path == "/api/correct":
            self.send_json(self._handle_correct(body))
        elif path == "/api/csv/save":
            self.send_json(self._handle_csv_save(body))
        elif path == "/api/benchmark/run":
            self.send_json(self._handle_benchmark_run(body))
        elif path == "/api/explain/word":
            self.send_json(self._handle_explain_word(body))
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data: Any, status: int = 200) -> None:
        """Send formatted JSON response."""
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> dict[str, Any]:
        """Read and parse incoming JSON payload."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # --- API Implementation Methods ---

    def _get_sys_info(self) -> dict[str, Any]:
        """Return runtime hardware, PyTorch, and Transformers status."""
        torch_info = "Not installed"
        device = "CPU"
        try:
            import torch
            torch_info = torch.__version__
            if torch.cuda.is_available():
                device = f"CUDA ({torch.cuda.get_device_name(0)})"
        except ImportError:
            pass

        transformers_info = "4.40+"
        try:
            import transformers
            transformers_info = transformers.__version__
        except ImportError:
            pass

        return {
            "torch_version": torch_info,
            "torch_device": device,
            "transformers_version": transformers_info,
            "python_version": sys.version.split()[0],
        }

    def _get_vocabulary(self) -> dict[str, Any]:
        """Return corpus vocabulary statistics and top frequency words."""
        freqs = _BASELINE.frequencies
        sorted_words = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
        return {
            "size": len(_BASELINE.vocabulary),
            "tokens": sum(freqs.values()),
            "top": [{"word": word, "freq": count} for word, count in sorted_words],
        }

    def _handle_explain_word(self, body: dict[str, Any]) -> dict[str, Any]:
        """Calculate detailed candidate ranking and confidence breakdown for one word."""
        word = body.get("word", "").strip()
        if not word:
            return {"error": "Empty word"}

        in_vocab = word.lower() in _BASELINE.vocabulary
        correction = _BASELINE.correct_word(word)
        query_soundex = _soundex(word)

        # Retrieve all candidate neighbors
        candidates = []
        for candidate in _BASELINE.vocabulary:
            if abs(len(candidate) - len(word.lower())) <= _BASELINE.max_distance:
                dist = _edit_distance(word.lower(), candidate)
                if dist <= _BASELINE.max_distance:
                    dam_dist = _damerau_levenshtein_distance(word.lower(), candidate)
                    cand_soundex = _soundex(candidate)
                    candidates.append({
                        "word": candidate,
                        "distance": dist,
                        "damerau_distance": dam_dist,
                        "soundex": cand_soundex,
                        "phonetic_match": cand_soundex == query_soundex,
                        "freq": _BASELINE.frequencies[candidate],
                    })

        candidates.sort(key=lambda x: (x["distance"], -x["freq"], x["word"]))

        return {
            "word": word,
            "soundex": query_soundex,
            "in_vocabulary": in_vocab,
            "corrected": correction.corrected,
            "distance": correction.distance,
            "damerau_distance": correction.damerau_distance,
            "phonetic_match": correction.phonetic_match,
            "confidence": round(correction.confidence, 4),
            "candidates": candidates[:15],
        }

    def _handle_correct(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run baseline NLP correction and optional LLMs, computing diffs and metrics."""
        text = body.get("text", "").strip()
        if not text:
            return {"error": "Input text is required"}

        max_distance = int(body.get("max_distance", 2))
        expected_text = body.get("expected_text", "").strip()
        model_names = body.get("models", [])

        corrector = (
            _BASELINE
            if max_distance == _BASELINE.max_distance
            else NoisyChannelSpellCorrector(max_distance=max_distance)
        )

        # Baseline correction
        t0 = time.perf_counter()
        baseline_text, raw_corrections = corrector.correct(text)
        baseline_latency = round((time.perf_counter() - t0) * 1000, 1)

        corrections_data = [
            {
                "original": c.original,
                "corrected": c.corrected,
                "distance": c.distance,
                "damerau_distance": c.damerau_distance,
                "phonetic_match": c.phonetic_match,
                "confidence": round(c.confidence, 3),
            }
            for c in raw_corrections
            if c.distance > 0
        ]

        # Calculate metrics if expected ground truth provided
        baseline_acc = 0.0
        baseline_prec = 0.0
        baseline_rec = 0.0
        baseline_f1 = 0.0
        baseline_cer = 0.0
        baseline_wer = 0.0
        baseline_exact = False

        if expected_text:
            baseline_acc = word_accuracy(expected_text, baseline_text)
            b_metrics = correction_metrics(text, expected_text, baseline_text)
            baseline_prec = b_metrics["precision"]
            baseline_rec = b_metrics["recall"]
            baseline_f1 = b_metrics["f1"]
            baseline_cer = character_error_rate(expected_text, baseline_text)
            baseline_wer = word_error_rate(expected_text, baseline_text)
            baseline_exact = sentence_exact_match(expected_text, baseline_text)

        # Run LLM Models
        llm_results = []
        for model_name in model_names:
            model_name = model_name.strip()
            if not model_name:
                continue

            t_start = time.perf_counter()
            try:
                llm = get_llm(model_name)
                llm_out = llm.correct(text)
                latency = round((time.perf_counter() - t_start) * 1000, 1)

                acc = 0.0
                prec = 0.0
                rec = 0.0
                f1 = 0.0
                cer = 0.0
                wer = 0.0
                exact = False
                if expected_text:
                    acc = word_accuracy(expected_text, llm_out)
                    m = correction_metrics(text, expected_text, llm_out)
                    prec = m["precision"]
                    rec = m["recall"]
                    f1 = m["f1"]
                    cer = character_error_rate(expected_text, llm_out)
                    wer = word_error_rate(expected_text, llm_out)
                    exact = sentence_exact_match(expected_text, llm_out)

                llm_results.append({
                    "model": model_name,
                    "status": "ok",
                    "text": llm_out,
                    "latency_ms": latency,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "cer": cer,
                    "wer": wer,
                    "exact_match": exact,
                })
            except Exception as exc:
                llm_results.append({
                    "model": model_name,
                    "status": "error",
                    "error": str(exc),
                    "text": "",
                    "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
                })

        return {
            "input_text": text,
            "expected_text": expected_text,
            "baseline_text": baseline_text,
            "baseline_latency_ms": baseline_latency,
            "baseline_accuracy": baseline_acc,
            "baseline_precision": baseline_prec,
            "baseline_recall": baseline_rec,
            "baseline_f1": baseline_f1,
            "baseline_cer": baseline_cer,
            "baseline_wer": baseline_wer,
            "baseline_exact_match": baseline_exact,
            "corrections": corrections_data,
            "llm_results": llm_results,
        }

    def _get_csv_list(self) -> list[dict[str, Any]]:
        """List all CSV files in workspace with metadata."""
        files = []
        for path in sorted(BASE_DIR.glob("*.csv")):
            try:
                size_kb = round(path.stat().st_size / 1024, 1)
                row_count = 0
                with path.open(newline="", encoding="utf-8-sig") as f:
                    row_count = max(0, sum(1 for _ in f) - 1)
                files.append({
                    "name": path.name,
                    "rows": row_count,
                    "size_kb": size_kb,
                    "modified": path.stat().st_mtime,
                })
            except Exception:
                pass
        return files

    def _read_csv_file(self, filename: str) -> dict[str, Any]:
        """Safely read and parse CSV content."""
        clean_name = Path(filename).name
        target = BASE_DIR / clean_name
        if not target.is_file():
            return {"headers": [], "rows": [], "error": "File not found"}

        try:
            with target.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = list(reader.fieldnames or [])
                rows = list(reader)
            return {"headers": headers, "rows": rows}
        except Exception as exc:
            return {"headers": [], "rows": [], "error": str(exc)}

    def _handle_csv_save(self, body: dict[str, Any]) -> dict[str, Any]:
        """Save comparison run record to a CSV file."""
        target_path = BASE_DIR / "spelling_correction_results.csv"
        fields = [
            "input_text",
            "expected_text",
            "detected_errors",
            "baseline_text",
            "baseline_accuracy",
            "baseline_precision",
            "baseline_recall",
            "baseline_f1",
            "llm_text",
            "llm_accuracy",
            "llm_precision",
            "llm_recall",
            "llm_f1",
            "llm_status",
        ]

        # Extract LLM summary
        llm_results = body.get("llm_results", [])
        llm_text = "\n\n".join(f"[{r['model']}] {r['text']}" for r in llm_results if r.get("status") == "ok")
        llm_status = "; ".join(f"{r['model']}: {r.get('status')}" for r in llm_results)

        llm_first = llm_results[0] if llm_results else {}
        row = {
            "input_text": body.get("input_text", ""),
            "expected_text": body.get("expected_text", ""),
            "detected_errors": "; ".join(c.get("original", "") for c in body.get("corrections", [])),
            "baseline_text": body.get("baseline_text", ""),
            "baseline_accuracy": f"{body.get('baseline_accuracy', 0):.3f}",
            "baseline_precision": f"{body.get('baseline_precision', 0):.3f}",
            "baseline_recall": f"{body.get('baseline_recall', 0):.3f}",
            "baseline_f1": f"{body.get('baseline_f1', 0):.3f}",
            "llm_text": llm_text,
            "llm_accuracy": f"{llm_first.get('accuracy', 0):.3f}" if llm_first else "",
            "llm_precision": f"{llm_first.get('precision', 0):.3f}" if llm_first else "",
            "llm_recall": f"{llm_first.get('recall', 0):.3f}" if llm_first else "",
            "llm_f1": f"{llm_first.get('f1', 0):.3f}" if llm_first else "",
            "llm_status": llm_status or "offline",
        }

        file_exists = target_path.is_file()
        with target_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return {"success": True, "filename": target_path.name}

    def _handle_benchmark_run(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute batch benchmark on a dataset file."""
        cases_file = Path(body.get("cases_file", "my_cases.csv")).name
        output_file = Path(body.get("output_file", "batch_eval_results.csv")).name

        cases_path = BASE_DIR / cases_file
        output_path = BASE_DIR / output_file
        models = body.get("models", [])

        cases = load_cases(cases_path)
        run_cli_cases(
            cases=cases,
            output=output_path,
            llm_model=None,
            max_distance=2,
            llm_models=models,
        )

        return {
            "success": True,
            "cases_count": len(cases),
            "output_file": output_path.name,
        }


def start_server(host: str = "127.0.0.1", port: int = 8765) -> HTTPServer:
    """Start the multi-threaded HTTP server in a background thread."""
    server = HTTPServer((host, port), LexiFixRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    """Application entry point with native desktop window and browser support."""
    parser = argparse.ArgumentParser(description="LexiFix AI — Spelling Intelligence Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765)")
    parser.add_argument("--browser", action="store_true", help="Launch in default browser instead of native window")
    parser.add_argument("--no-open", action="store_true", help="Start server without launching window or browser")
    args = parser.parse_args()

    port = find_free_port(args.port)
    url = f"http://{args.host}:{port}/"
    server = start_server(args.host, port)

    print("\n" + "=" * 65)
    print("   ✦ LEXIFIX AI — SPELLING INTELLIGENCE STUDIO v2.0")
    print(f"   ⚡ Local Server: {url}")
    print("   ✓ Deterministic NLP Baseline: Active & Offline")
    print("   ✓ Transformer Models: Ready")
    print("=" * 65 + "\n")

    if args.no_open:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down LexiFix AI server...")
            server.shutdown()
        return

    # Attempt native window launch with pywebview
    use_webview = not args.browser
    if use_webview:
        try:
            import webview  # type: ignore

            window = webview.create_window(
                "LexiFix AI — Spelling Intelligence Studio",
                url=url,
                width=1280,
                height=860,
                min_size=(960, 680),
                text_select=True,
            )
            webview.start()
            server.shutdown()
            return
        except Exception as exc:
            print(f"[Notice] Desktop window could not initialize ({exc}). Falling back to web browser...")

    # Fallback to default browser
    import webbrowser
    webbrowser.open(url)
    print(f"Opened LexiFix AI in default browser: {url}")
    print("Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down LexiFix AI server...")
        server.shutdown()


if __name__ == "__main__":
    main()
