"""Compatibility wrapper for the renamed actor keyboard player."""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("play_actor_keyboard.py")), run_name="__main__")
