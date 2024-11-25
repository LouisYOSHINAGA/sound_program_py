import sys
sys.path.append("..")
from instruments.winds import saxophone
from instruments.utils import render_pad_save

render_pad_save(saxophone, save_title="p1006_saxophone.wav",
                note_no=49, velocity=100, gate=1.0, duration=2.0)