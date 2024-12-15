import sys
sys.path.append("..")
from instruments.keys import harpsichord
from instruments.utils import render_pad_save

render_pad_save(harpsichord, save_title="p1203_harpsichord.wav",
                note_no=48, velocity=100, gate=4.0, duration=5.0)