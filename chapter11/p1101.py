import sys
sys.path.append("..")
from instruments.strings import violin
from instruments.utils import render_pad_save

render_pad_save(violin, save_title="p1101_violin.wav",
                note_no=60, velocity=100, gate=1.0, duration=2.0)