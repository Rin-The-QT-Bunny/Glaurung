'''
Title: Program Synthesis of Dalek Mind
Author: Celerinsil Meleko
Date: September 19 2021
'''

import pyautogui
import pyperclip
pyautogui.PAUSE = 0.1  #pause for 1 second
pyautogui.FAILSAFE = True  # Automatic intiate problem solved

def screen_config():
    return pyautogui.size()

def position():
    return print(pyautogui.position())
# Control the mouth
 
def press_enter():
    pyautogui.keyDown('enter')  #按下回车键
    pyautogui.keyUp('enter')  #释放回车键

def backspace():
    pyautogui.press('backspace')
 
def scroll(x):
    pyautogui.scroll(x) #向上滚动200

def write_program(p):
    pyautogui.typewrite(p,0.001) 

program = "def test(x):\nx = x +1 \nx = x * x \nreturn x \n"
tr  = "print(test(3))\nprint('Testing Completed')"

write_program(program)
backspace()
write_program(tr)

