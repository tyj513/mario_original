from js import (
    document,
    setInterval,
)

from pyodide.ffi import create_proxy

import pyxel
from level import Level
import settings

class App:

    def __init__(self):
        canvasDOM = document.querySelector("#canvas")
        
        # initialize ctx
        pyxel.init(256, 200, canvasDOM)
        pyxel.load_assets(["https://raw.githubusercontent.com/Barrarroso/mariopyscript/main/assets/tiles.png", "https://raw.githubusercontent.com/Barrarroso/mariopyscript/main/assets/spritesheet_mario.png", "https://raw.githubusercontent.com/Barrarroso/mariopyscript/main/assets/background_03.png"])
        
        self.level = Level(settings.level01)
        self.start_game()
    
    def start_game(self):
        proxy = create_proxy(self.game_loop)
        setInterval(proxy, 33, "a parameter");

    def game_loop(self, *args, **kwargs):
        #requestAnimationFrame(create_proxy(self.game_loop))
        self.update()
        self.draw()

    def update(self):
        pyxel.update()
        
        if not pyxel.loading:
            self.level.update()



    def draw(self):
        pyxel.cls()
        
        if not pyxel.loading:
            self.level.draw()
        else:
            pyxel.text(100, 100, "Loading...", 7)

App()