import sys
sys.path.append("..")
from instruments.brasses import trumpet
from instruments.utils import render_pad_save

render_pad_save(trumpet, save_title="p1007_trumpet.wav",
                note_no=58, velocity=100, gate=1.0, duration=2.0)