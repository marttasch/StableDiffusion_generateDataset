import os
import time

def get_TimeElapsed(startTime):
    endTime = time.time()
    timeElapsed = endTime - startTime
    hours, remainder = divmod(timeElapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    timeElapsedStr = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    return timeElapsedStr