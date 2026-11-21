"""
Small window for running mosaic2d without the command line.

    mosaic2d gui            (or double-click mosaic2d.exe)

Choose the photo folder, an output folder and a few options, press Run. The
log appears in the window; the work runs in a background thread with the same
code as ``mosaic2d run``.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

RESOLUTIONS = ["1.0", "0.8", "0.5", "0.25"]
PROFILES = ["auto", "rgb", "agrowing6", "multiband"]
ACCURACIES = ["medium", "high", "highest", "low"]


class _QueueWriter:
    """File-like object that forwards text to a queue (log window)."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s)

    def flush(self):
        pass


def build_argv(inp, out, resolution, profile, accuracy, tiles, name="mosaic", height=None,
               sequence=False, lens_from=None, ref_band=None, quick=False):
    argv = ["run", str(inp), "--resolution", str(resolution), "--profile", profile,
            "--accuracy", accuracy, "--name", name]
    if height:
        argv += ["--height", str(height)]
    if out:
        argv += ["-o", str(out)]
    if tiles:
        argv += ["--tiles", "8192"]
    if sequence:
        argv += ["--sequence"]
    if lens_from:
        argv += ["--lens-from", str(lens_from)]
    if ref_band:
        argv += ["--ref-band", str(ref_band)]
    if quick:
        argv += ["--quick"]
    return argv


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        from . import __version__
        self.tk, self.ttk, self.root = tk, ttk, root
        root.title(f"mosaic2d {__version__}")
        root.minsize(640, 420)
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.out_dir: Path | None = None

        self.v_in = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_name = tk.StringVar(value="mosaic")
        self.v_res = tk.StringVar(value="1.0")
        self.v_prof = tk.StringVar(value="auto")
        self.v_acc = tk.StringVar(value="medium")
        self.v_tiles = tk.BooleanVar(value=False)
        self.v_height = tk.StringVar(value="")
        self.v_seq = tk.BooleanVar(value=False)
        self.v_lens = tk.StringVar(value="")
        self.v_ref = tk.StringVar(value="")
        self.v_quick = tk.BooleanVar(value=False)
        self.v_status = tk.StringVar(value="Choose the folder with the photos.")

        f = ttk.Frame(root, padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        r = 0
        ttk.Label(f, text="Photo folder").grid(row=r, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_in).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Browse...", command=self.pick_in).grid(row=r, column=2)
        r += 1
        ttk.Label(f, text="Output folder").grid(row=r, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_out).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Browse...", command=self.pick_out).grid(row=r, column=2)
        r += 1
        ttk.Label(f, text="Lens from (optional)").grid(row=r, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_lens).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Browse...", command=self.pick_lens).grid(row=r, column=2)
        r += 1
        opts = ttk.Frame(f)
        opts.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 4))
        for label, var, values, width in (("Name", self.v_name, None, 12),
                                          ("Resolution", self.v_res, RESOLUTIONS, 6),
                                          ("Camera", self.v_prof, PROFILES, 10),
                                          ("Accuracy", self.v_acc, ACCURACIES, 8)):
            ttk.Label(opts, text=label).pack(side="left", padx=(0, 4))
            if values is None:
                ttk.Entry(opts, textvariable=var, width=width).pack(side="left", padx=(0, 12))
            else:
                ttk.Combobox(opts, textvariable=var, values=values, width=width).pack(side="left", padx=(0, 12))
        ttk.Label(opts, text="Ref. band").pack(side="left", padx=(0, 4))
        ttk.Entry(opts, textvariable=self.v_ref, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(opts, text="Height (m)").pack(side="left", padx=(0, 4))
        ttk.Entry(opts, textvariable=self.v_height, width=6).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="8192 px tiles + .vrt", variable=self.v_tiles).pack(side="left")
        r += 1
        ttk.Checkbutton(f, text="Photo series (sequence): file names in shooting order - match neighbours "
                               "harder and bridge gaps between consecutive photos",
                        variable=self.v_seq).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Checkbutton(f, text="Quick: SIFT only and fewer extra pairs (much faster matching)",
                        variable=self.v_quick).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        bar = ttk.Frame(f)
        bar.grid(row=r, column=0, columnspan=3, sticky="ew", pady=4)
        self.b_run = ttk.Button(bar, text="Run", command=self.start)
        self.b_run.pack(side="left")
        self.b_open = ttk.Button(bar, text="Open output folder", command=self.open_out, state="disabled")
        self.b_open.pack(side="left", padx=6)
        ttk.Button(bar, text="View a mosaic...", command=self.view).pack(side="left", padx=6)
        self.pb = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.pb.pack(side="left", padx=6)
        ttk.Label(bar, textvariable=self.v_status).pack(side="left", padx=6)
        r += 1
        f.rowconfigure(r, weight=1)
        self.log = tk.Text(f, height=16, wrap="none", font=("Consolas", 9))
        self.log.grid(row=r, column=0, columnspan=3, sticky="nsew")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.log.yview)
        sb.grid(row=r, column=3, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)
        root.after(150, self.poll)

    # --- actions ---------------------------------------------------------
    def pick_in(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Folder with the photos")
        if d:
            self.v_in.set(d)
            if not self.v_out.get():
                self.v_out.set(str(Path(d) / "mosaic2d"))
            self.v_name.set(Path(d).name or "mosaic")

    def pick_out(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.v_out.set(d)

    def pick_lens(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Output folder of an earlier run (same camera and housing)")
        if d:
            self.v_lens.set(d)

    def start(self):
        from tkinter import messagebox
        inp = self.v_in.get().strip()
        if not inp or not Path(inp).is_dir():
            messagebox.showerror("mosaic2d", "Choose an existing photo folder first.")
            return
        try:
            res = float(self.v_res.get())
            assert 0 < res <= 4
        except (ValueError, AssertionError):
            messagebox.showerror("mosaic2d", "Resolution must be a number such as 1.0 or 0.5.")
            return
        height = None
        if self.v_height.get().strip():
            try:
                height = float(self.v_height.get().replace(",", "."))
                assert height > 0
            except (ValueError, AssertionError):
                messagebox.showerror("mosaic2d", "Height must be the camera-to-ground distance in metres, "
                                                 "e.g. 1.5 (or leave it empty).")
                return
        out = self.v_out.get().strip() or str(Path(inp) / "mosaic2d")
        self.out_dir = Path(out)
        argv = build_argv(inp, out, res, self.v_prof.get(), self.v_acc.get(), self.v_tiles.get(),
                          self.v_name.get().strip() or "mosaic", height, self.v_seq.get(),
                          self.v_lens.get().strip() or None, self.v_ref.get().strip() or None,
                          self.v_quick.get())
        self.log.delete("1.0", "end")
        self.q.put("mosaic2d " + " ".join(argv) + "\n\n")
        self.b_run.configure(state="disabled")
        self.b_open.configure(state="disabled")
        self.pb.start(12)
        self.v_status.set("Running... (this can take a while)")
        self.worker = threading.Thread(target=self._work, args=(argv,), daemon=True)
        self.worker.start()

    def _work(self, argv):
        from .cli import main
        old = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _QueueWriter(self.q)
        try:
            code = main(argv)
            self.q.put(("done", code))
        except SystemExit as e:                       # argparse errors
            self.q.put(("done", e.code if isinstance(e.code, int) else 2))
        except Exception as e:                        # show the error, keep the window
            import traceback
            self.q.put(traceback.format_exc())
            self.q.put(("error", str(e)))
        finally:
            sys.stdout, sys.stderr = old

    def poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    self.finish(*item)
                else:
                    self.log.insert("end", item)
                    self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def finish(self, kind, info):
        self.pb.stop()
        self.b_run.configure(state="normal")
        if kind == "done" and info in (0, None):
            self.v_status.set("Finished.")
        elif kind == "done":
            self.v_status.set("Finished; not all images could be aligned (see the report).")
        else:
            self.v_status.set(f"Error: {info}")
        if self.out_dir and self.out_dir.exists():
            self.b_open.configure(state="normal")

    def view(self):
        """Viewer for any (multiband) TIFF; starts in the output folder if there is one."""
        from tkinter import filedialog, messagebox
        start = self.out_dir if self.out_dir and self.out_dir.exists() else (self.v_out.get() or None)
        fn = filedialog.askopenfilename(title="Mosaic to view", initialdir=start or None,
                                        filetypes=[("TIFF", "*.tif *.tiff"), ("all files", "*.*")])
        if not fn:
            return
        from .viewer import ViewerWindow
        try:
            ViewerWindow(self.root, fn)
        except Exception as e:                        # noqa: BLE001
            messagebox.showerror("mosaic2d viewer", f"Cannot open {Path(fn).name}:\n{e}")

    def open_out(self):
        if not self.out_dir:
            return
        if sys.platform.startswith("win"):
            os.startfile(self.out_dir)  # noqa: S606 (opens Explorer)
        else:
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(self.out_dir)])


def main(selftest: bool = False) -> int:
    import tkinter as tk
    root = tk.Tk()
    App(root)
    try:                                              # bring the window in front of the console
        root.lift()
        root.attributes("-topmost", True)
        root.after(800, lambda: root.attributes("-topmost", False))
        root.focus_force()
    except Exception:
        pass
    if selftest:                                      # build the window once and close (CI check)
        root.update()
        root.destroy()
        print("gui ok")
        return 0
    root.mainloop()
    return 0
