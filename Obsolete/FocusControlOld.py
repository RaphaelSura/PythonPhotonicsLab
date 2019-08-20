import sys
sys.path.append("..")
from tkinter import Tk, Entry, Label, Button, END, StringVar, CENTER
from LowLevelModules.XPScontroller import XPSstage
app_font = ('Latex', 20)


class FocusApp:

    def __init__(self, name):
        self.win = Tk()
        self.win.resizable(1, 1)
        self.win.title(name)

        # stage related stuff
        self.cursor = 1
        self.pos = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        ip_address = '169.254.66.147'
        portnum = 5001
        groupnum = 1
        self.stage = XPSstage(ip_address, portnum, groupnum)
        print(self.stage.read_position())

        # buttons and label
        instruction = "\n Use ←/→ to move the objective back and forth \n Use ↑/↓ to toggle between step size \n"
        top_label = Label(self.win, text=instruction, font=('Latex', 14))
        top_label.grid(row=0, column=0, columnspan=4, pady=5)
        self.entry_value = StringVar()
        self.entry = Entry(self.win, textvariable=self.entry_value, width=10, font=app_font, justify=CENTER)
        self.entry_value.set(str(self.pos[self.cursor]))
        self.entry.grid(row=2, column=0, padx=5, pady=5)
        micron_lbl = Label(self.win, text="μm      ", height=3, width=10, font=app_font)
        micron_lbl.grid(row=2, column=1, padx=5, pady=5)
        left_bt = Button(self.win, text="←", command=self.move_backward, height=3, width=10, font=app_font)
        left_bt.grid(row=2, column=2, padx=5, pady=5)
        right_bt = Button(self.win, text="→", command=self.move_forward, height=3, width=10, font=app_font)
        right_bt.grid(row=2, column=3, padx=5, pady=5)
        up_bt = Button(self.win, text="↑", command=self.increment_up, height=3, width=10, font=app_font)
        up_bt.grid(row=1, column=0, padx=5, pady=5)
        down_bt = Button(self.win, text="↓", command=self.increment_down, height=3, width=10, font=app_font)
        down_bt.grid(row=3, column=0, padx=5, pady=5)

        self.win.bind('<Up>', self.increment_up)
        self.win.bind('<Down>', self.increment_down)
        self.win.bind('<Left>', self.move_backward)
        self.win.bind('<Right>', self.move_forward)

    def update_entry(self):
        self.entry.delete(0, END)
        self.entry.insert(0, self.pos[self.cursor])

    def move_backward(self, _event=None):
        self.stage.move_by(float(self.entry_value.get()) / 1000)

    def move_forward(self, _event=None):
        self.stage.move_by(-1 * float(self.entry_value.get()) / 1000)

    def increment_up(self, _event=None):
        if self.cursor < len(self.pos) - 1:
            self.cursor += 1
            self.update_entry()

    def increment_down(self, _event=None):
        if self.cursor > 0:
            self.cursor -= 1
            self.update_entry()


app = FocusApp("Confocal focus control")
app.win.mainloop()
