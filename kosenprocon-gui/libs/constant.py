import configparser

# check
IS_LOCAL = True
API_URL = r"https://procon33-practice.kosen.work"
PROXY = {
    "http": "",
    "https": "",
}
#

SRC_SAMPLE_RATE = 48000
COMPRESSING_RATE = 16
BASE_AUDIO_DIR = r"samples/JKspeech"
LOAD_BASE_NUM = 88

TIMEOUT = 3.0
LOG_LEVEL = "DEBUG"

EXAMPLE_PROBLEM = r"samples/original/problem4.wav"

FILTER_THRESHOLD = 10

DLL_DIR = r"./build"


cfg = configparser.ConfigParser(defaults={"ACCESS_TOKEN": "xxxxx"})
cfg.read("./config.ini")
API_TOKEN = cfg.get("DEFAULT", "ACCESS_TOKEN")