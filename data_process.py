#!coding=utf8
__author__ = 'fangruiyu'
import os
import sys
import codecs

cbt_dir = u"D:\\语料文件\\CTB5.0\\DATA\\POSTAGGED"
training_file = "D:\\data\\structure perceptron\\data\\cbt_5_training.file"
test_file = "D:\\data\\structure perceptron\\data\\cbt_5_test.file"

def gen_training_data(cbt_dir, training_file, test_file):
    if not os.path.isdir(cbt_dir):
        return
    training_writer = codecs.open(training_file, "w", "utf-8")
    test_writer = codecs.open(test_file, "w", "utf-8")
    for file in os.listdir(cbt_dir):
        in_reader = codecs.open(cbt_dir + "\\" + file, "r", "gb2312")
        try:
            for line in in_reader.readlines():
                line = line.strip()
                if line.startswith("<"):
                    continue
                units = line.split(u"，_PU")
                for unit in units:
                    training_writer.write(unit.replace("_", "/") + "\n")
        except Exception, e:
            print e
            continue
    training_writer.close()
    test_writer.close()

if __name__ == "__main__":
    gen_training_data(cbt_dir, training_file, test_file)