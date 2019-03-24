from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
# from LowLevelModules.XPScontroller import XPSstage


class Container(GridLayout):
    display = ObjectProperty()
    # stage related stuff
    cursor = 2
    position = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    # stage = XPSstage('169.254.66.147', 5001, 1)

    def __init__(self, **kwargs):
        super(Container, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(None, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _on_keyboard_down(self, kb, keycode, *_):
        if keycode[1] == 'up':
            self.increment_up()
        if keycode[1] == 'down':
            self.increment_down()
        if keycode[1] == 'left':
            self.move_stage_backward()
        if keycode[1] == 'right':
            self.move_stage_forward()
        return True

    def update_entry(self):
        self.display.text = str(self.position[self.cursor])

    def increment_up(self):
        if self.cursor < len(self.position) - 1:
            self.cursor += 1
            self.update_entry()

    def increment_down(self):
        if self.cursor > 0:
            self.cursor -= 1
            self.update_entry()

    def move_stage_forward(self):
        # self.stage.move_by(-1 * float(self.display.text) / 1000)
        pass

    def move_stage_backward(self):
        # self.stage.move_by(float(self.display.text) / 1000)
        pass


class FocuscontrolApp(App):

    def build(self):
        self.title = 'Focus control'
        self.icon = "focus_app_icon.png"
        return Container()


FocuscontrolApp().run()   # must have a focuscontrol.kv in same folder (base name in lower case minus App)
