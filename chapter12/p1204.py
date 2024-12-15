import sys
sys.path.append("..")
from instruments.keys import acoustic_piano
from instruments.utils import render_pad_save

render_pad_save(acoustic_piano, save_title="p1204_acoustic_piano.wav",
                note_no=48, velocity=100, gate=4.0, duration=5.0)