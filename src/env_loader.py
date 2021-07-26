# Loads environment variables
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(Path().absolute(), ".env"))
# print(Path().absolute())

print(os.environ)
logging.basicConfig(level=int(os.environ.get('LOG_LEVEL')))


def string_arr_to_path(string):

    if string.find(",") == -1:
        return string
    else:
        returnStr = ""
        strArray = str(string).split(",")

        for i in strArray:
            returnStr = os.path.join(returnStr, i)

        return returnStr
