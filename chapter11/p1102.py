import sys
sys.path.append("..")
from instruments.strings import viola
from instruments.utils import render_pad_save

render_pad_save(viola, save_title="p1102_viola.wav",
                note_no=60, velocity=100, gate=1.0, duration=2.0)