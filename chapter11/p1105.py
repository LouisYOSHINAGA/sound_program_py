import sys
sys.path.append("..")
from instruments.strings import harp 
from instruments.utils import render_pad_save

render_pad_save(harp, save_title="p1105_harp.wav",
                note_no=60, velocity=100, gate=4.0, duration=5.0)