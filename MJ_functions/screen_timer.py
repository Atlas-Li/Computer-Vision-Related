# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:53:34 2022

@author: Mingjun Li
"""

import turtle as t
import random
from datetime import datetime

t.setup(1280, 720)

while True:
    t.colormode(255)
    t.color(random.randint(0, 255),
          random.randint(0, 255),
          random.randint(0, 255))
    t.begin_fill()
    t.penup()
    t.setpos(t.pos()[0], t.pos()[1])
    t.pendown()
    t.clear()
    t.write(datetime.now().strftime('%H:%M:%S.%f')[:-3], 
            move=False,
            align="center",
            font=("Times",150,"bold"))
t.done()
