import sys
sys.path.append("..")
from instruments.strings import contrabass
from instruments.utils import render_pad_save

render_pad_save(contrabass, save_title="p1104_contrabass.wav",
                note_no=48, velocity=100, gate=1.0, duration=2.0)