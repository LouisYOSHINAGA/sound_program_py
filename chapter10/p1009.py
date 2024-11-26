import sys
sys.path.append("..")
from instruments.brasses import horn
from instruments.utils import render_pad_save

render_pad_save(horn, save_title="p1009_horn.wav",
                note_no=48, velocity=100, gate=1.0, duration=2.0)