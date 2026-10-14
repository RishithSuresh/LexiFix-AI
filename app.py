"""Desktop user interface for LexiFix AI.

The interface deliberately uses Tkinter from Python's standard library, so
the project can be used without installing a separate web framework.
"""

from __future__ import annotations

import csv
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from spelling_correction import (
    LLMCorrector,
    NoisyChannelSpellCorrector,
)


class LexiFixApp(tk.Tk):
    """Main window that connects the user interface to the correction models."""

    def __init__(self) -> None:
        super().__init__()
        self.title("LexiFix AI - Spelling Correction")
        self.geometry("1100x760")
        self.minsize(850, 600)

        self.baseline = NoisyChannelSpellCorrector()
        self.llm: LLMCorrector | None = None
        self.last_result: dict[str, str] | None = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create the layout and keep the controls grouped by workflow."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(4, weight=1)

        header = ttk.Frame(self, padding=16)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="LexiFix AI",
            font=("Segoe UI", 22, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Detect, correct, and compare spelling with Python NLP and an optional LLM.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        options = ttk.LabelFrame(self, text="Model settings", padding=10)
        options.grid(row=1, column=0, padx=16, sticky="ew")
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="LLM model (optional):").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="vennify/t5-base-grammar-correction")
        ttk.Entry(options, textvariable=self.model_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Label(options, text="The first LLM run may download the model.").grid(
            row=0, column=2, sticky="w"
        )

        input_frame = ttk.LabelFrame(self, text="Input text", padding=10)
        input_frame.grid(row=2, column=0, padx=16, pady=(12, 8), sticky="nsew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        self.input_text = tk.Text(input_frame, height=8, wrap="word", undo=True)
        self.input_text.grid(row=0, column=0, sticky="nsew")
        self.input_text.insert(
            "1.0",
            "I recieve the mesage from my frend.",
        )
        input_scroll = ttk.Scrollbar(
            input_frame, orient="vertical", command=self.input_text.yview
        )
        input_scroll.grid(row=0, column=1, sticky="ns")
        self.input_text.configure(yscrollcommand=input_scroll.set)

        actions = ttk.Frame(self, padding=(16, 0))
        actions.grid(row=3, column=0, sticky="ew")
        self.correct_button = ttk.Button(
            actions, text="Correct and compare", command=self._start_correction
        )
        self.correct_button.pack(side="left")
        ttk.Button(actions, text="Clear", command=self._clear).pack(side="left", padx=8)
        ttk.Button(actions, text="Save result as CSV", command=self._save_csv).pack(
            side="left"
        )
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        output = ttk.Frame(self, padding=16)
        output.grid(row=4, column=0, sticky="nsew")
        output.columnconfigure(0, weight=1)
        output.columnconfigure(1, weight=1)
        output.rowconfigure(1, weight=1)
        ttk.Label(output, text="Python NLP baseline", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(output, text="LLM correction", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        self.baseline_output = self._make_output_box(output, 0)
        self.llm_output = self._make_output_box(output, 1)

        self.details_var = tk.StringVar(value="Run a correction to see detected errors and metrics.")
        ttk.Label(
            self, textvariable=self.details_var, padding=(16, 0, 16, 12), wraplength=1050
        ).grid(row=5, column=0, sticky="w")

    def _make_output_box(self, parent: ttk.Frame, column: int) -> tk.Text:
        """Create one read-only-looking text area for a model result."""
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 12, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text = tk.Text(frame, height=10, wrap="word", state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        return text

    def _set_output(self, widget: tk.Text, value: str) -> None:
        """Replace a result box's contents while keeping it non-editable."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _start_correction(self) -> None:
        """Validate input and run model work away from the UI thread."""
        source = self.input_text.get("1.0", "end-1c").strip()
        if not source:
            messagebox.showwarning("Input required", "Enter text to correct first.")
            return
        model_name = self.model_var.get().strip()
        self.correct_button.configure(state="disabled")
        self.status_var.set("Running baseline correction...")
        threading.Thread(
            target=self._correct_in_background,
            args=(source, model_name),
            daemon=True,
        ).start()

    def _correct_in_background(self, source: str, model_name: str) -> None:
        """Run correction without freezing the window during an LLM download."""
        try:
            baseline_text, corrections = self.baseline.correct(source)
            llm_text = ""
            llm_status = "not configured"
            if model_name:
                if self.llm is None or self.llm.model_name != model_name:
                    self.llm = LLMCorrector(model_name)
                llm_text = self.llm.correct(source)
                llm_status = "ok"
            result = {
                "input_text": source,
                "expected_text": "",
                "detected_errors": "; ".join(
                    item.original for item in corrections if item.distance > 0
                ),
                "baseline_text": baseline_text,
                "baseline_accuracy": "",
                "baseline_precision": "",
                "baseline_recall": "",
                "baseline_f1": "",
                "llm_text": llm_text,
                "llm_accuracy": "",
                "llm_precision": "",
                "llm_recall": "",
                "llm_f1": "",
                "llm_status": llm_status,
            }
            self.after(0, self._show_result, result)
        except (RuntimeError, OSError, ValueError) as error:
            self.after(0, self._show_error, str(error))

    def _show_result(self, result: dict[str, str]) -> None:
        """Render model outputs and explain the baseline's detected changes."""
        self.last_result = result
        self._set_output(self.baseline_output, result["baseline_text"])
        self._set_output(
            self.llm_output,
            result["llm_text"] or "LLM disabled. Clear the model field to use baseline only.",
        )
        corrections = result["detected_errors"] or "No baseline spelling errors detected."
        details = f"Detected errors: {corrections}"
        if result["llm_status"] == "ok":
            details += "\nThe LLM result is shown for comparison. Use Save result as CSV to export this run."
        else:
            details += "\nLLM status: not configured."
        self.details_var.set(details)
        self.status_var.set("Completed")
        self.correct_button.configure(state="normal")

    def _show_error(self, message: str) -> None:
        """Show an actionable model error and restore the button."""
        self.status_var.set("Error")
        self.correct_button.configure(state="normal")
        messagebox.showerror("Correction failed", message)

    def _clear(self) -> None:
        """Reset the editor and result panels."""
        self.input_text.delete("1.0", "end")
        self._set_output(self.baseline_output, "")
        self._set_output(self.llm_output, "")
        self.details_var.set("Run a correction to see detected errors and metrics.")
        self.status_var.set("Ready")
        self.last_result = None

    def _save_csv(self) -> None:
        """Save the current correction as a row compatible with the CLI CSV."""
        if self.last_result is None:
            messagebox.showinfo("Nothing to save", "Run a correction before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Save correction result",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        fields = list(self.last_result)
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(self.last_result)
        self.status_var.set(f"Saved: {Path(path).name}")


if __name__ == "__main__":
    LexiFixApp().mainloop()
