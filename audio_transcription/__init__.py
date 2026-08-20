# Offline audio transcription and translation for English, Urdu and Hindi.

# Imported lazily so that config, audio, models and pipeline stay importable without tkinter.
__all__ = ["GuiComponent", "ProcessingPipeline"]


def __getattr__(name):
    if name == "GuiComponent":
        from .gui import GuiComponent
        return GuiComponent
    if name == "ProcessingPipeline":
        from .pipeline import ProcessingPipeline
        return ProcessingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
