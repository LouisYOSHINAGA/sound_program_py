import sys
sys.path.append("..")
from effect import distortion
from instruments.strings import electric_guitar
from instruments.utils import render_pad_save

render_pad_save(electric_guitar, save_title="p1107_electric_guitar.wav",
                note_no=60, velocity=100, gate=4.0, duration=5.0)

render_pad_save(electric_guitar, save_title="p1107_electric_guitar_distort.wav",
                note_no=60, velocity=100, gate=4.0, duration=5.0,
                post_effect=distortion, gain=1000, level=0.2)