import sys
sys.path.append("..")
from instruments.strings import electric_bass
from instruments.utils import render_pad_save

render_pad_save(electric_bass, save_title="p1108_electric_bass.wav",
                note_no=28, velocity=100, gate=4.0, duration=5.0)