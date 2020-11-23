# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     file
   Description :
   Author :       ybw
   date：          2020/11/16
-------------------------------------------------
   Change Activity:
                   2020/11/16:
-------------------------------------------------
"""
import os


def mkdir(path):
    """
    保证只创建一次
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def getFile(dir_path):
    dict_file = {}
    List_Files = []
    for path, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(path, file)
            size = os.path.getsize(file_path)
            dict_file[file_path] = size
            List_Files.append(file_path)
    return List_Files, dict_file


