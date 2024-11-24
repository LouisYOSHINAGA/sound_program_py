import sys
sys.path.append("..")
from instruments.winds import flute
from instruments.utils import render_pad_save

render_pad_save(flute, save_title="p1001_flute.wav",
                note_no=60, velocity=100, gate=1.0, duration=2.0)