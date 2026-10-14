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

MODEL_OPTIONS = (
    "vennify/t5-base-grammar-correction",
    "vennify/t5-small-grammar-correction",
    "google/flan-t5-small",
)

COLORS = {
    "background": "#0b1120",
    "sidebar": "#111a2e",
    "surface": "#151f35",
    "surface_dark": "#1d2942",
    "text": "#f4f7ff",
    "muted": "#8e9bb5",
    "primary": "#7c6cff",
    "primary_dark": "#6657e8",
    "success": "#39d98a",
    "border": "#263452",
    "input": "#10192c",
}


class LexiFixApp(tk.Tk):
    """Main window that connects the user interface to the correction models."""

    def __init__(self) -> None:
        super().__init__()
        self.title("LexiFix AI - Spelling Correction")
        self.geometry("1180x820")
        self.minsize(900, 680)
        self.configure(background=COLORS["background"])
        self._configure_theme()

        self.baseline = NoisyChannelSpellCorrector()
        self.llms: dict[str, LLMCorrector] = {}
        self.last_result: dict[str, str] | None = None
        self.csv_rows: list[dict[str, str]] = []
        self._build_widgets()
        self._load_latest_csv()

    def _configure_theme(self) -> None:
        """Define a restrained visual theme using only Tkinter primitives."""
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "App.TFrame",
            background=COLORS["background"],
        )
        style.configure(
            "Card.TLabelframe",
            background=COLORS["surface"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Primary.TButton",
            background=COLORS["primary"],
            foreground="white",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["primary_dark"]), ("disabled", "#9db5e8")],
        )
        style.configure(
            "Secondary.TButton",
            background=COLORS["surface_dark"],
            foreground=COLORS["text"],
            borderwidth=0,
            padding=(12, 9),
        )
        style.configure(
            "App.TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
        )
        style.configure(
            "Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=30,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["surface_dark"],
            foreground=COLORS["text"],
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Treeview",
            background=[("selected", COLORS["primary"])],
            foreground=[("selected", "white")],
        )

    def _build_widgets(self) -> None:
        """Create the branded shell and switchable product views."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        shell = tk.Frame(self, background=COLORS["background"])
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        sidebar = tk.Frame(shell, background=COLORS["sidebar"], width=230)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        ttk.Label(
            sidebar, text="LEXIFIX", foreground=COLORS["primary"],
            background=COLORS["sidebar"], font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", padx=24, pady=(32, 2))
        ttk.Label(
            sidebar, text="AI SPELLING STUDIO", foreground=COLORS["muted"],
            background=COLORS["sidebar"], font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=25, pady=(0, 40))

        self.nav_buttons: dict[str, tk.Button] = {}
        self._add_nav_button(sidebar, "workspace", "Correction workspace", "01")
        self._add_nav_button(sidebar, "csv", "Comparison results", "02")
        tk.Frame(sidebar, background=COLORS["border"], height=1).pack(
            fill="x", padx=24, pady=28
        )
        ttk.Label(
            sidebar, text="LOCAL & EXPLAINABLE", foreground=COLORS["muted"],
            background=COLORS["sidebar"], font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=25)
        ttk.Label(
            sidebar,
            text="Offline baseline\nOptional Transformer models\nCSV-ready evaluation",
            foreground="#b7c2d9", background=COLORS["sidebar"],
            font=("Segoe UI", 9), justify="left",
        ).pack(anchor="w", padx=25, pady=(10, 0))
        ttk.Label(
            sidebar, text="v1.0  •  LexiFix AI", foreground="#596984",
            background=COLORS["sidebar"], font=("Segoe UI", 8),
        ).pack(side="bottom", anchor="w", padx=25, pady=24)

        content = tk.Frame(shell, background=COLORS["background"])
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.view_container = content
        correction_tab = tk.Frame(content, background=COLORS["background"])
        csv_tab = tk.Frame(content, background=COLORS["background"])
        self.views = {"workspace": correction_tab, "csv": csv_tab}
        for view in self.views.values():
            view.grid(row=0, column=0, sticky="nsew")
        self._build_correction_tab(correction_tab)
        self._build_csv_tab(csv_tab)
        self._show_view("workspace")

    def _add_nav_button(self, parent: tk.Frame, key: str, label: str, number: str) -> None:
        """Add a custom navigation control instead of a native tab strip."""
        button = tk.Button(
            parent, text=f"  {number}    {label}", anchor="w",
            command=lambda: self._show_view(key),
            background=COLORS["sidebar"], foreground=COLORS["muted"],
            activebackground=COLORS["surface_dark"], activeforeground=COLORS["text"],
            relief="flat", bd=0, padx=16, pady=12,
            font=("Segoe UI", 10), cursor="hand2",
        )
        button.pack(fill="x", padx=12, pady=3)
        self.nav_buttons[key] = button

    def _show_view(self, key: str) -> None:
        """Raise a product view and highlight its navigation item."""
        self.views[key].tkraise()
        for name, button in self.nav_buttons.items():
            active = name == key
            button.configure(
                background=COLORS["surface_dark"] if active else COLORS["sidebar"],
                foreground=COLORS["text"] if active else COLORS["muted"],
                font=("Segoe UI", 10, "bold") if active else ("Segoe UI", 10),
            )

    def _build_correction_tab(self, parent: ttk.Frame) -> None:
        """Build the text correction workspace inside the first tab."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(4, weight=1)

        header = ttk.Frame(parent, padding=(28, 24, 28, 16), style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="LexiFix AI",
            font=("Segoe UI", 26, "bold"),
            foreground=COLORS["text"],
            background=COLORS["background"],
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Detect, correct, and compare spelling with Python NLP and an optional LLM.",
            foreground=COLORS["muted"],
            background=COLORS["background"],
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Label(
            header,
            text="SPELLING INTELLIGENCE",
            foreground=COLORS["primary"],
            background=COLORS["background"],
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=1, sticky="e")

        options = ttk.LabelFrame(
            parent, text="  Model settings  ", padding=(16, 12), style="Card.TLabelframe"
        )
        options.grid(row=1, column=0, padx=28, sticky="ew")
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Compare models:", style="App.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.model_list = tk.Listbox(
            options, height=3, selectmode="multiple", exportselection=False,
            font=("Segoe UI", 9), relief="flat", bd=0,
            background="#fbfcfe", foreground=COLORS["text"],
        )
        self.model_list.grid(row=0, column=1, sticky="ew", padx=8)
        for model in MODEL_OPTIONS:
            self.model_list.insert("end", model)
        self.model_list.selection_set(0, 1)
        ttk.Label(
            options,
            text="Select one or more\nmodels to compare",
            foreground=COLORS["success"],
            background=COLORS["surface"],
            font=("Segoe UI", 9, "bold"),
        ).grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        ttk.Label(options, text="Custom model (optional):", style="App.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        self.custom_model_var = tk.StringVar()
        ttk.Entry(options, textvariable=self.custom_model_var).grid(
            row=1, column=1, sticky="ew", padx=8, pady=(10, 0)
        )
        ttk.Label(
            options, text="Leave blank to use presets", foreground=COLORS["muted"],
            background=COLORS["surface"], font=("Segoe UI", 9)
        ).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(10, 0))

        input_frame = ttk.LabelFrame(
            parent, text="  Input text  ", padding=(16, 14), style="Card.TLabelframe"
        )
        input_frame.grid(row=2, column=0, padx=28, pady=(16, 10), sticky="nsew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        self.input_text = tk.Text(
            input_frame,
            height=8,
            wrap="word",
            undo=True,
            font=("Segoe UI", 11),
            padx=10,
            pady=10,
            relief="flat",
            borderwidth=0,
            background=COLORS["input"],
            foreground=COLORS["text"],
            insertbackground=COLORS["primary"],
        )
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

        actions = ttk.Frame(parent, padding=(28, 2, 28, 8), style="App.TFrame")
        actions.grid(row=3, column=0, sticky="ew")
        self.correct_button = ttk.Button(
            actions, text="✦  Correct and compare", command=self._start_correction,
            style="Primary.TButton"
        )
        self.correct_button.pack(side="left")
        ttk.Button(actions, text="Clear", command=self._clear, style="Secondary.TButton").pack(
            side="left", padx=8
        )
        ttk.Button(
            actions, text="Save result as CSV", command=self._save_csv,
            style="Secondary.TButton"
        ).pack(
            side="left"
        )
        self.status_var = tk.StringVar(value="●  Ready")
        ttk.Label(
            actions, textvariable=self.status_var, foreground=COLORS["muted"],
            background=COLORS["background"], font=("Segoe UI", 9, "bold")
        ).pack(side="right")

        output = ttk.Frame(parent, padding=(28, 4, 28, 0), style="App.TFrame")
        output.grid(row=4, column=0, sticky="nsew")
        output.columnconfigure(0, weight=1)
        output.columnconfigure(1, weight=1)
        output.rowconfigure(1, weight=1)
        ttk.Label(
            output, text="  PYTHON NLP BASELINE", font=("Segoe UI", 10, "bold"),
            foreground=COLORS["primary"], background=COLORS["background"]
        ).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            output, text="  LLM CORRECTION", font=("Segoe UI", 10, "bold"),
            foreground=COLORS["primary"], background=COLORS["background"]
        ).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        self.baseline_output = self._make_output_box(output, 0)
        self.llm_output = self._make_output_box(output, 1)

        self.baseline_meta_var = tk.StringVar(value="Deterministic • Offline")
        self.llm_meta_var = tk.StringVar(value="Optional • Model not run")
        ttk.Label(
            output, textvariable=self.baseline_meta_var, foreground=COLORS["muted"],
            background=COLORS["background"], font=("Segoe UI", 9)
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(
            output, textvariable=self.llm_meta_var, foreground=COLORS["muted"],
            background=COLORS["background"], font=("Segoe UI", 9)
        ).grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(6, 0))
        self.details_var = tk.StringVar(
            value="Run a correction to see detected errors and model details."
        )
        ttk.Label(
            parent, textvariable=self.details_var, padding=(28, 12, 28, 18),
            wraplength=1120, foreground=COLORS["muted"],
            background=COLORS["background"], font=("Segoe UI", 9)
        ).grid(row=5, column=0, sticky="w")

    def _build_csv_tab(self, parent: ttk.Frame) -> None:
        """Build a table for inspecting generated comparison CSV files."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(parent, padding=(28, 24, 28, 12), style="App.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(
            toolbar,
            text="Comparison results",
            font=("Segoe UI", 18, "bold"),
            foreground=COLORS["text"],
            background=COLORS["background"],
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            toolbar, text="Open CSV file", command=self._open_csv,
            style="Secondary.TButton",
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            toolbar, text="Refresh", command=self._load_latest_csv,
            style="Secondary.TButton",
        ).grid(row=0, column=2, padx=(8, 0))

        table_frame = tk.Frame(
            parent, background=COLORS["surface"],
            highlightbackground=COLORS["border"], highlightthickness=1, bd=0,
        )
        table_frame.grid(row=1, column=0, padx=28, pady=(0, 18), sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.csv_table = ttk.Treeview(table_frame, show="headings")
        self.csv_table.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.csv_table.yview
        )
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.csv_table.xview
        )
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.csv_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.csv_status_var = tk.StringVar(value="No CSV loaded")
        ttk.Label(
            parent, textvariable=self.csv_status_var, padding=(28, 0, 28, 12),
            foreground=COLORS["muted"], background=COLORS["background"],
        ).grid(row=2, column=0, sticky="w")

    def _load_latest_csv(self) -> None:
        """Load the most recent comparison CSV from the project directory."""
        candidates = [
            Path("multi_model_results.csv"),
            Path("llm_comparison_results.csv"),
            Path("my_results.csv"),
            Path("spelling_correction_results.csv"),
        ]
        for path in candidates:
            if path.is_file():
                self._load_csv(path)
                return
        self.csv_status_var.set("No comparison CSV found. Run a comparison first.")

    def _open_csv(self) -> None:
        """Let the user select any comparison CSV to display."""
        path = filedialog.askopenfilename(
            title="Open comparison CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._load_csv(Path(path))

    def _load_csv(self, path: Path) -> None:
        """Read CSV rows and render them with horizontal and vertical scrolling."""
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError("The CSV contains no data rows.")
            columns = list(rows[0])
            self.csv_rows = rows
            self.csv_table.configure(columns=columns)
            for column in columns:
                self.csv_table.heading(column, text=column)
                self.csv_table.column(column, width=max(130, min(360, len(column) * 10)))
            for item in self.csv_table.get_children():
                self.csv_table.delete(item)
            for row in rows:
                self.csv_table.insert("", "end", values=[row.get(column, "") for column in columns])
            self.csv_status_var.set(f"{path.name} • {len(rows)} result(s)")
        except (OSError, csv.Error, ValueError) as error:
            messagebox.showerror("CSV could not be loaded", str(error))

    def _make_output_box(self, parent: ttk.Frame, column: int) -> tk.Text:
        """Create one read-only-looking text area for a model result."""
        frame = tk.Frame(
            parent, background=COLORS["surface"], highlightbackground=COLORS["border"],
            highlightthickness=1, bd=0
        )
        frame.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 12, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text = tk.Text(
            frame, height=10, wrap="word", state="disabled",
            font=("Segoe UI", 11), padx=12, pady=12, relief="flat", bd=0,
            background=COLORS["surface"], foreground=COLORS["text"],
        )
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
        selected = [self.model_list.get(index) for index in self.model_list.curselection()]
        custom_model = self.custom_model_var.get().strip()
        if custom_model and custom_model not in selected:
            selected.append(custom_model)
        self.correct_button.configure(state="disabled")
        self.status_var.set("●  Processing...")
        threading.Thread(
            target=self._correct_in_background,
            args=(source, selected),
            daemon=True,
        ).start()

    def _correct_in_background(self, source: str, model_names: list[str]) -> None:
        """Run correction without freezing the window during an LLM download."""
        try:
            baseline_text, corrections = self.baseline.correct(source)
            llm_outputs: list[str] = []
            statuses: list[str] = []
            for model_name in model_names:
                if model_name not in self.llms:
                    self.llms[model_name] = LLMCorrector(model_name)
                try:
                    llm_outputs.append(
                        f"[{model_name}]\n{self.llms[model_name].correct(source)}"
                    )
                    statuses.append(f"{model_name}: ok")
                except (RuntimeError, OSError, ValueError) as error:
                    statuses.append(f"{model_name}: error: {error}")
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
                "llm_text": "\n\n".join(llm_outputs),
                "llm_accuracy": "",
                "llm_precision": "",
                "llm_recall": "",
                "llm_f1": "",
                "llm_status": "; ".join(statuses) if statuses else "not configured",
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
        self.status_var.set("●  Completed")
        self.baseline_meta_var.set(
            f"Deterministic • {len(result['detected_errors'].split('; ')) if result['detected_errors'] else 0} corrections"
        )
        self.llm_meta_var.set(
            "Optional • Completed" if result["llm_status"] == "ok"
            else "Optional • Not configured"
        )
        self.correct_button.configure(state="normal")

    def _show_error(self, message: str) -> None:
        """Show an actionable model error and restore the button."""
        self.status_var.set("●  Error")
        self.correct_button.configure(state="normal")
        messagebox.showerror("Correction failed", message)

    def _clear(self) -> None:
        """Reset the editor and result panels."""
        self.input_text.delete("1.0", "end")
        self._set_output(self.baseline_output, "")
        self._set_output(self.llm_output, "")
        self.details_var.set("Run a correction to see detected errors and model details.")
        self.baseline_meta_var.set("Deterministic • Offline")
        self.llm_meta_var.set("Optional • Model not run")
        self.status_var.set("●  Ready")
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
        self._load_csv(Path(path))
        self._show_view("csv")
        self.status_var.set(f"Saved: {Path(path).name}")


if __name__ == "__main__":
    LexiFixApp().mainloop()
