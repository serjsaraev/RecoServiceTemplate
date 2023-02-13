from dotenv import load_dotenv
from environs import Env

load_dotenv()
env = Env()


class Settings:
    SPACE_NAME = env.str("SPACE_NAME", 'negdotprod')
    NUM_THREADS = env.int("NUM_THREADS", 8)
    EFS = env.int("EFS", 128)
    EFC = env.int("EFC", 256)
    M = env.int("M", 256)
