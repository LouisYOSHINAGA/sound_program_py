import sys
sys.path.append("..")
from instruments.winds import clarinet
from instruments.utils import render_pad_save

render_pad_save(clarinet, save_title="p1003_clarinet.wav",
                note_no=50, velocity=100, gate=1.0, duration=2.0)