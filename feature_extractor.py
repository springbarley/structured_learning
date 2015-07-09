__author__ = 'fangruiyu'
import numpy as np
import sys
import os
import codecs
import re

class FTemplates:
    '''
    templates consist of template tuples
    template tuples is a sequence list of wi:token and pi:label
    wi denotes current word
    wi-1 denotes previous word
    wi+1 denotes next word
    pi denotes current label
    pi+1 denotes next label
    wi&pi+1 denotes sequences of current and next label if exists
    '''
    #based on kaixu zhang EACL segment joint pos tag features
    zkx_templates = [
        ('wi-1&pi'), ('wi&pi'), ('wi+1&pi'), ('wi-2&wi-1&pi'), ('wi-1&wi&pi'),
        ('wi&wi+1&pi'), ('wi+1&wi+2&pi'), ('pi-1&pi')
    ]
    line_begin = "<"
    line_end = ">"
    indexs_set = None
    '''
    parse a template string, generate indexs for both words and labels
    '''
    @staticmethod
    def parseTemplate(template_string):
        indexs = []
        split = "&"
        units = template_string.split("&")
        for unit in units:
            prefix = unit[0]
            if units[1:] == "i":
                indexs.append(prefix + ":" + 0 + split)
            else:
                if units[2] != '+' or units[2] != '-':
                    raise Exception("template string format error: ", unit, " in ", template_string)
                else:
                    if units[2] == '+':
                        indexs.append(prefix + ":" + int(units[3:]) + split)
                    elif units[2] == '-':
                        indexs.append(prefix + ":" + 0 - int(units[3:]) + split)
                    else:
                        raise Exception("template string format error: ", unit, " in ", template_string)
        return indexs

    @staticmethod
    def parseTemplates():

        if FTemplates.indexs_set != None:
            return FTemplates.indexs_set
        FTemplates.indexs_set = []
        for template_string in FTemplates.zkx_templates:
            index = []
            split = "&"
            units = template_string.split("&")
            for unit in units:
                prefix = unit[0]
                if unit[1:] == "i":
                    index.append(prefix + ":" + str(0));
                else:
                    if unit[2] != '+' and unit[2] != '-':
                        raise Exception("template string format error: ", unit, " in ", template_string)
                    else:
                        if unit[2] == '+':
                            index.append(prefix + ":" + (unit[3:]))
                        elif unit[2] == '-':
                            index.append(prefix + ":" + str(0 - int(unit[3:])))
                        else:
                            raise Exception("template string format error: ", unit, " in ", template_string)
            FTemplates.indexs_set.append(index)
        return FTemplates.indexs_set

class FeatureExtractor:
    '''
get tokens\labels list from given line
'''
    label_set_ = set([])
    @staticmethod
    def getRawLabeledSequence(line, tokens, labels):
        units = re.split("\\s+", line.strip())
        for unit in units:
            tl = unit.split("/")
            if len(tl) != 2:
                print ("unit: ", tl , "error! in line" )
                continue
            else:
                tokens.append(tl[0])
                labels.append(tl[1])

    @staticmethod
    def extractInstanceFeatures(tokens, labels):
        features = {}
        for pos in range(0, len(tokens)):
            #iterate on multi feature templates
            for indexs in FTemplates.parseTemplates():
                a_feature = ""
                try:
                    '''
                        parse a feature_template_string
                    '''
                    for slice in indexs:
                        if slice.startswith("w"):
                            type = "w"
                            temp = tokens
                        else:
                            type = "p"
                            temp = labels
                        index = int(slice[2:])
                        if pos + index in range(0, len(temp)):
                            a_feature += (type + str(index) + ":" + temp[pos+index] + "|")
                        #if index out of boundary, add line begin or end simbol instead, but
                        #boundary type (tokens or labels) should be specified.
                        elif pos + index < 0:
                            a_feature += (type + str(index) + ":" + FTemplates.line_begin + "|")
                        elif pos + index >= len(temp):
                            a_feature += (type + str(index) + ":" + FTemplates.line_end + "|")
                            #
                        else:
                            raise IndexError("index error of: " + slice)
                except Exception, e:
                    print e
                if not features.has_key(a_feature):
                    features[a_feature] = 1
                else:
                    features[a_feature] = features[a_feature] + 1
        return features

    @staticmethod
    def add_new_weights(features, weights):
        for key, value in features.iteritems():
            if not weights.has_key(key):
                weights[key] = 0.0

    @staticmethod
    def extractInstanceFeaturesOnPosition(tokens, labels, position):
        features = {}

        #iterate on multi feature templates
        for indexs in FTemplates.parseTemplates():
            a_feature = ""
            try:
                '''
                    parse a feature_template_string
                '''
                for slice in indexs:
                    type = ""
                    temp = []
                    if slice.startswith("w"):
                        type = "w"
                        temp = tokens
                    else:
                        type = "p"
                        temp = labels
                    index = int(slice[2:])
                    if position + index in range(0, len(temp)):
                        a_feature += (type + str(index) + ":" + temp[position+index] + "|")
                    #if index out of boundary, add line begin or end simbol instead, but
                    #boundary type (tokens or labels) should be specified.
                    elif position + index < 0:
                        a_feature += (type + str(index) + ":" + FTemplates.line_begin + "|")
                    elif position + index >= len(temp):
                        a_feature += (type + str(index) + ":" + FTemplates.line_end + "|")
                        #
                    else:
                        raise IndexError("index error of: " + slice)
            except Exception, e:
                print e
            if not features.has_key(a_feature):
                features[a_feature] = 1
            else:
                features[a_feature] = features[a_feature] + 1
        return features

    @staticmethod
    def gen_feature_map_and_labels(in_file, feature_map_out, label_out):
        print "Generate feature map and labels ..."
        in_reader = codecs.open(in_file, "r", "utf-8")
        fm_out = codecs.open(feature_map_out, "w", "utf-8")
        label_set_out = codecs.open(label_out, "w", "utf-8")
        feature_total = {}
        num = 0
        for line in in_reader.readlines():
            num += 1
            if num % 999 == 0:
                print ".",
            try:
                tokens = []
                labels = []
                FeatureExtractor.getRawLabeledSequence(line, tokens, labels)
                #add labels
                for label in labels:
                    FeatureExtractor.label_set_.add(label)

                features = FeatureExtractor.extractInstanceFeatures(tokens, labels)
                for (key, value) in features.iteritems():
                    if feature_total.has_key(key):
                        feature_total[key] = feature_total[key] + features[key]
                    else:
                        feature_total[key] = features[key]
            except Exception, e:
                print e

        for key, value in feature_total.iteritems():
            fm_out.write(key + "\t" + str(value) + "\n")
        for label in FeatureExtractor.label_set_:
            label_set_out.write(label + "\n")
        fm_out.close()
        label_set_out.close()
        print "Generate feature map and labels done ..."
    @staticmethod
    def load_feature_index(feature_index_file):
        feature_index = {}
        feature_reader = codecs.open(feature_index_file, "r", "utf-8")
        pos = 0
        for line in feature_reader.readlines():
            units = line.strip().split("\t")
            feature_index[units[0]] = pos
            pos += 1
        return feature_index

    @staticmethod
    def load_labels(label_file):
        label_set = []
        label_reader = codecs.open(label_file, "r", "utf-8")
        for line in label_reader.readlines():
            label_set.append(line.strip())
        return label_set

    @staticmethod
    def index_features(features, feature_index):
        indexed_features = np.zeros(len(feature_index))
        for key, value in features.iteritems():
            if feature_index.has_key(key):
                indexed_features[feature_index[key]] = value
            else:
                continue
                #raise IndexError("feature: ", key, " not exists in feature index file!")
        return indexed_features

if __name__ == "__main__":
    #original input file
    input = "D:\\data\\structure perceptron\\sp-train.txt"
    feature_map = "D:\\data\\structure perceptron\\fm.txt"
    labels = "D:\\data\\structure perceptron\\labels.txt"
    FeatureExtractor.gen_feature_map_and_labels(input, feature_map, labels)