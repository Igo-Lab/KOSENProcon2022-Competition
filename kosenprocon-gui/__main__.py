import os

from . import libs
from .MainScreen import App

if __name__ == "__main__":
    os.environ["DISPLAY"] = ":0.0"
    App.app()
