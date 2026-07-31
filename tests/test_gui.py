import os
import sys
import warnings
from tkinter import ttk

from PIL import ImageGrab

from dna2graph.constants import APP_NAME
from dna2graph.gui.main_window import MainWindow


def _find_button(parent, text):
    for widget in parent.winfo_children():
        if isinstance(widget, ttk.Button) and widget.cget("text") == text:
            return widget

        button = _find_button(widget, text)
        if button is not None:
            return button

    return None


def _capture_window(window, output_path):
    try:
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        window.focus_force()
        window.update_idletasks()
        window.update()
        window.after(250)
        window.update()

        left = window.winfo_rootx()
        top = window.winfo_rooty()
        right = left + window.winfo_width()
        bottom = top + window.winfo_height()

        grab_options = {}
        if "DISPLAY" in os.environ:
            grab_options["xdisplay"] = os.environ["DISPLAY"]

        screenshot = ImageGrab.grab(
            bbox=(left, top, right, bottom),
            **grab_options,
        )
        screenshot.save(output_path)
        window.attributes("-topmost", False)
    except Exception as error:
        warnings.warn(
            f"Unable to capture {output_path.name} on {sys.platform}: "
            f"{error}",
            RuntimeWarning,
            stacklevel=1,
        )


def test_gui_opens(tmp_path):
    window = MainWindow()

    try:
        window.update_idletasks()
        window.update()

        assert window.winfo_exists()
        assert window.title() == APP_NAME

        advanced_button = _find_button(window, "Advanced")
        assert advanced_button is not None

        advanced_button.invoke()
        window.update_idletasks()
        window.update()

        advanced_window = window.advanced_window
        assert advanced_window is not None
        assert advanced_window.winfo_exists()
        assert advanced_window.title() == "Advanced Preferences"

        _capture_window(
            advanced_window,
            tmp_path / "dna2graph_advanced_preferences.png",
        )

        cancel_button = _find_button(advanced_window, "Cancel")
        assert cancel_button is not None

        cancel_button.invoke()
        window.update_idletasks()
        window.update()

        assert not advanced_window.winfo_exists()
        _capture_window(window, tmp_path / "dna2graph_gui.png")
    finally:
        window.destroy()
