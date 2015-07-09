__author__ = 'fangruiyu'
import feature_extractor as fe
import numpy as np

class LabelNode:
    def __init__(self, position, label):
        self.position_ = position
        self.max_antecedent_ = None
        self.max_subsequences_ = None
        self.label_ = label
        self.score_ = -1;
    def add_max_antecedent(self, node):
        self.max_antecedent_ = node

    def add_subsequences(self, node):
        self.max_subsequences_.append(node)
    def set_score(self, score):
        self.score_ = score
    def get_score(self):
        return self.score_
    def get_ancestors(self):
        result = []
        temp = self
        result.append(temp.label_)
        while temp.max_antecedent_ != None:
            result.append(temp.max_antecedent_.label_)
            temp = temp.max_antecedent_
        return list(reversed(result))

class SeqNode:
    def __init__(self, position, tokens):
        self.position_ = position
        self.antecedent_ = None
        self.sebsequences_ = None
        self.labels_ = []
    def add_label(self, label_node):
        self.labels_.append(label_node)
    def get_labels(self):
        return self.labels_
'''
data structure to represent decoding lattice
'''
class lattice:
    def __init__(self, tokens):
        self.seq_nodes_ = []
        self.tokens_ = tokens
    def add_seq_node(self, seq_nodes):
        self.seq_nodes_.append(seq_nodes)

    def get_label_nodes(self, position):
        if position < 0 or position > len(self.tokens_):
            raise IndexError("positon ", position, " out of boundary!")
        return self.seq_nodes_[position].get_labels()

class decoder:
    def __init__(self, labels):
        self.labels_ = labels

    def multiply(self, features, weights):
        score = 0.0
        for key, value in features.iteritems():
            if weights.has_key(key):
                score += features[key] * weights[key]
            else:
                continue
        return score

    '''
     assign each token with all possible labels using FTemplates and current weights
     generate a latticed
    '''
    def do_train_decode(self, tokens, weights):
        self.tokens_ = tokens
        self.lattice_ = lattice(tokens)
        for pos in range(0, len(self.tokens_)):
            seq_node = SeqNode(pos, self.tokens_)
            max_label = None
            #iterate on all possible labels
            for possible_label in self.labels_:
                history_labels = []
                #get labels of last position seq node, if pos >= 1
                if pos >= 1:
                    max_score = 0
                    max_previous_label = None
                    #iterate on previous position's labels, pick possible_label with maximum score on history condition
                    for pre_label_node in self.lattice_.get_label_nodes(pos - 1):
                        previous_score = pre_label_node.get_score()
                        #fetch previous assigned labels
                        history_labels =  pre_label_node.get_ancestors()
                        history_labels.append(possible_label)
                        newly_features = fe.FeatureExtractor.extractInstanceFeaturesOnPosition(self.tokens_, history_labels, pos)
                        #add newly extracted features, initial weight is zero
                        fe.FeatureExtractor.add_new_weights(newly_features, weights)
                        '''
                        for token in tokens:
                            print token,
                        print
                        for label in history_labels:
                            print label,
                        print
                        for key, value in newly_features.iteritems():
                            print "(" + key + "," + str(value) + ")",
                        print
                        '''
                        incremented_score = self.multiply(newly_features, weights)
                        total_score = previous_score + incremented_score
                        if total_score >= max_score:
                            max_score = total_score
                            max_previous_label = pre_label_node
                    max_possible_label = LabelNode(pos, possible_label)
                    max_possible_label.add_max_antecedent(max_previous_label)
                    max_possible_label.set_score(max_score)
                    seq_node.add_label(max_possible_label)
                else:
                    history_labels.append(possible_label)
                    newly_features = fe.FeatureExtractor.extractInstanceFeaturesOnPosition(self.tokens_, history_labels, pos)
                    #add weights for newly extracted features, initial value is zero
                    fe.FeatureExtractor.add_new_weights(newly_features, weights)
                    incremented_score = self.multiply(newly_features, weights)
                    max_possible_label = LabelNode(pos, possible_label)
                    max_possible_label.set_score(incremented_score)
                    seq_node.add_label(max_possible_label)
            self.lattice_.add_seq_node(seq_node)

    def do_decode(self, tokens, weights):
        self.tokens_ = tokens
        self.lattice_ = lattice(tokens)
        for pos in range(0, len(self.tokens_)):
            seq_node = SeqNode(pos, self.tokens_)
            max_label = None
            #iterate on all possible labels
            for possible_label in self.labels_:
                history_labels = []
                #get labels of last position seq node, if pos >= 1
                if pos >= 1:
                    max_score = 0
                    max_previous_label = None
                    #iterate on previous position's labels, pick possible_label with maximum score on history condition
                    for pre_label_node in self.lattice_.get_label_nodes(pos - 1):
                        previous_score = pre_label_node.get_score()
                        #fetch previous assigned labels
                        history_labels =  pre_label_node.get_ancestors()
                        history_labels.append(possible_label)
                        newly_features = fe.FeatureExtractor.extractInstanceFeaturesOnPosition(self.tokens_, history_labels, pos)
                        '''
                        for token in tokens:
                            print token,
                        print
                        for label in history_labels:
                            print label,
                        print
                        for key, value in newly_features.iteritems():
                            print "(" + key + "," + str(value) + ")",
                        print
                        '''
                        incremented_score = self.multiply(newly_features, weights)
                        total_score = previous_score + incremented_score
                        #print "total_score: ", total_score
                        if total_score >= max_score:
                            max_score = total_score
                            max_previous_label = pre_label_node
                    max_possible_label = LabelNode(pos, possible_label)
                    max_possible_label.add_max_antecedent(max_previous_label)
                    max_possible_label.set_score(max_score)
                    seq_node.add_label(max_possible_label)
                    #print "max ance for last label: ", possible_label, max_possible_label.get_ancestors(), " score: ", max_score
                else:
                    history_labels.append(possible_label)
                    newly_features = fe.FeatureExtractor.extractInstanceFeaturesOnPosition(self.tokens_, history_labels, pos)
                    '''
                    for key, value in newly_features.iteritems():
                        print "(" + key + "," + str(value) + ")",
                    print ""
                    '''
                    incremented_score = self.multiply(newly_features, weights)
                    max_possible_label = LabelNode(pos, possible_label)
                    max_possible_label.set_score(incremented_score)
                    seq_node.add_label(max_possible_label)
            self.lattice_.add_seq_node(seq_node)

    def get_assigned_labels(self):
        if len(self.lattice_.seq_nodes_) == 0:
            raise "lattice contains no decoding result"
        else:
            last_position = len(self.lattice_.seq_nodes_)
            max_score = 0
            max_label_node = None
            for label in self.lattice_.seq_nodes_[last_position - 1].get_labels():
                if label.get_score() >= max_score:
                    max_score = label.get_score()
                    max_label_node = label
            return max_label_node.get_ancestors()

#input: tokens to be labeled
#output: labels that maxmiums W*F(tokens, labels)
#using current weights to get the best labels
    def train_decode(self, tokens, weights):
        #decode in sgd train
        self.do_train_decode(tokens, weights)
        return self.get_assigned_labels()

    def viterbi_decode(self, tokens, weights):
        #decode without update weights
        self.do_decode(tokens, weights)
        return self.get_assigned_labels()