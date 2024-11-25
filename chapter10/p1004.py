import sys
sys.path.append("..")
from instruments.winds import oboe
from instruments.utils import render_pad_save

render_pad_save(oboe, save_title="p1004_oboe.wav",
                note_no=58, velocity=100, gate=1.0, duration=2.0)