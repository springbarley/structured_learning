__author__ = 'fangruiyu'
import codecs
import feature_extractor as fe
import viterbi_decoder as vdecoder

'''
input file format:
one instance per line, in format like that:
one instance per line, in format like that:
raw1/label1 raw2/label1 ...

generate feature map:
feature1: index1
feature2: index2
...
    '''
def update_weights(weights, assigned_features, correct_features, step, log):
    for key, value in weights.iteritems():
        assigned_num = 0
        correct_num = 0
        if assigned_features.has_key(key):
            assigned_num = assigned_features[key]
        if correct_features.has_key(key):
            correct_num = correct_features[key]
        #log.write("correct feature weights: " + key + " from: " + str(weights[key]) + " to ")
        weights[key] = value + (correct_num - assigned_num)
        #log.write(str(weights[key]) + "\n")

def sgd_train(label_file ,in_file, model_file):
    print "In GP training process ..."
    # firstly, feature_map and labels should be gen
    in_reader = codecs.open(in_file, "r", "utf-8")
    log = codecs.open("D:\\data\\structure perceptron\\log", "w", "utf-8")
    label_set = fe.FeatureExtractor.load_labels(label_file)
    model_writer = codecs.open(model_file, "w", "utf-8")

    weights = {}
    decoder = vdecoder.decoder(label_set)
    max_iter = 30
    pos = 0
    step = 1
    threshold = 0.003
    last_precision = 0
    while pos <= max_iter:
        wr_num = 0
        total_num = 0
        print "In iteration: %d..."%(pos)
        line_num = 0
        for line in in_reader.readlines():
            line_num += 1
            if line_num % 1000 == 0:
                print "Having decoded %d phrases ..."%(line_num)
            tokens = []
            labels = []
            fe.FeatureExtractor.getRawLabeledSequence(line.strip(), tokens, labels)
            correct_features = fe.FeatureExtractor.extractInstanceFeatures(tokens, labels)
            #add new feature weights to weights when encounter new features
            fe.FeatureExtractor.add_new_weights(correct_features, weights)

            assigned_labels = decoder.train_decode(tokens, weights)
            if ' '.join(assigned_labels) != ' '.join(labels):
                log.write(line + "\n")
                log.write("iter: " + str(pos) + "\n")
                log.write("correct labels: " + " ".join(labels) + "\n")
                log.write("decode labels:" + " ".join(assigned_labels) + "\n")
                wr_num += 1
                assigned_features = fe.FeatureExtractor.extractInstanceFeatures(tokens, assigned_labels)
                '''
                log.write("assigned features: ")
                for feature, value in assigned_features.iteritems():
                    log.write("(" + feature + "\t" + str(value) + ") ")
                log.write("\n")
                for key, value in weights.iteritems():
                    log.write("(" + key + "," + str(value) + ")")
                log.write("\n")
                '''
                update_weights(weights, assigned_features, correct_features, step, log)
                '''
                for key, value in weights.iteritems():
                    log.write("(" + key + "," + str(value) + ")")
                log.write("\n")
                '''
            total_num += 1
        precision = float(total_num - wr_num) / total_num
        print("Iteration: %d, get precision: %f"%(pos, precision))
        #if newly precision improves tightly, than break
        if precision - last_precision <= threshold:
            break
        last_precision = precision
        #if all instance are right
        if wr_num == 0:
            break
        pos += 1

        in_reader = codecs.open(in_file, "r", "utf-8")

    for key, value in weights.iteritems():
        if value != 0.0:
            model_writer.write(key + "\t" + str(value) + "\n")

    in_reader.close()
    model_writer.close()
    log.close()
def load_model(model_file):
    model_reader = codecs.open(model_file, "r", "utf-8")
    weights = {}
    for line in model_reader.readlines():
        units = line.strip().split("\t")
        weights[units[0]] = float(units[1])
    return weights

if __name__ == "__main__":
    '''
    ar = np.arange(10).reshape(2,5)
    print "************"
    print ar[..., np.newaxis]
    print "************"
    print ar[np.newaxis, ...]
    print "************"
    print ar[:, np.newaxis, :]
        '''

    '''
    sp_train = "D:\\data\\structure perceptron\\data\\cbt_5_training_sample.file"
    feature_map = "D:\\data\\structure perceptron\\fm.txt"
    labels = "D:\\data\\structure perceptron\\labels.txt"
    sp_model = "D:\\data\\structure perceptron\\cbt_5_sample.model"
    fe.FeatureExtractor.gen_feature_map_and_labels(sp_train, feature_map, labels)
    sgd_train(labels, sp_train, sp_model)
    '''

    '''test code
    '''

    sp_model = "D:\\data\\structure perceptron\\cbt_5_sample.model"
    sp_train = "D:\\data\\structure perceptron\\data\\cbt_5_test_sample.file"
    model = load_model(sp_model)
    in_reader = codecs.open(sp_train, "r", "utf-8")
    label_file = "D:\\data\\structure perceptron\\labels.txt"
    label_set = fe.FeatureExtractor.load_labels(label_file)
    ac_num = 0
    wr_num = 0
    total_num = 0
    for line in in_reader.readlines():
        total_num += 1
        tokens = []
        labels = []
        fe.FeatureExtractor.getRawLabeledSequence(line.strip(), tokens, labels)
        decoder = vdecoder.decoder(label_set)
        assigned_labels = decoder.viterbi_decode(tokens, model)
        if "".join(assigned_labels) == "".join(labels):
            ac_num += 1
            print "ac number: ", ac_num,
        print "output: ", assigned_labels
    print "precision: ", float(ac_num / total_num)
