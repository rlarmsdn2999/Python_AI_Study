import tkinter as tk
import random as rd

root = tk.TK()
root.geometry('200x100')

lbl = tk.Label(text = 'Label')
btn = tk.Button(text = 'Push')

lbl.pack()
btn.pack()
tk.mainloop()