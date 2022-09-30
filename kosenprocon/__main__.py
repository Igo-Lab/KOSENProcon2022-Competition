from . import libs
from .App import App

if __name__ == "__main__":
    logger = libs.main_logger.setLogger()
    App.app(logger)
