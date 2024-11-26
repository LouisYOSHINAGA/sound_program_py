import sys
sys.path.append("..")
from instruments.brasses import trombone
from instruments.utils import render_pad_save

render_pad_save(trombone, save_title="p1008_trombone.wav",
                note_no=34, velocity=100, gate=1.0, duration=2.0)