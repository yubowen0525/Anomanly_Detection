# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     time
   Description :
   Author :       ybw
   date：          2020/11/20
-------------------------------------------------
   Change Activity:
                   2020/11/20:
-------------------------------------------------
"""
import time

def getTime():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())