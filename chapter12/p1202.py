import sys
sys.path.append("..")
from instruments.keys import read_organ
from instruments.utils import render_pad_save

render_pad_save(read_organ, save_title="p1202_read_organ.wav",
                note_no=48, velocity=100, gate=1.0, duration=2.0)