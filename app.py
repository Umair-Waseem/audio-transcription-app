# Entry point; the application itself lives in the audio_transcription package.

import sys

import tkinter as tk

from audio_transcription import GuiComponent, ProcessingPipeline


def main() -> int:
    try:
        gui = GuiComponent()
    except tk.TclError as exc:
        print(
            "Failed to start the Tkinter GUI. Verify this Python installation with "
            "`python -m tkinter`, and reinstall Python with Tcl/Tk support if that fails.",
            file=sys.stderr,
        )
        print(f"Tkinter error: {exc}", file=sys.stderr)
        return 1

    gui.pipeline = ProcessingPipeline(gui)
    gui.build_controls()
    gui.root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
