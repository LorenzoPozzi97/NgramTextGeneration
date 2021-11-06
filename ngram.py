import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import random

import numpy as np
train_file = "ted-train.txt"

def read(fname, max_lines=np.inf):
    """
    Reads in the data in fname and returns it as
    one long list of words. Also returns a vocabulary in
    the form of a word2index and index2word dictionary.
    """
    data = []
    # w2i will automatically keep a counter to asign to new words
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    start = "<s>"
    end = "</s>"
    
    with open(fname, "r") as fh:
        for k, line in enumerate(fh):
            if k > max_lines:
                break
            words = line.strip().split()
            # assign an index to each word
            for w in words:
                i2w[w2i[w]] = w # trick
            
            sent = [start] + words + [end]
            data.append(sent)

    return data, w2i, i2w
   
   
def train_ngram(corpus, N, k=0):
    start_time = time.time()
    """
    Trains an n-gram language model with optional add-k smoothing
    and additionaly returns the unigram model
    
    :param data: text-data as returned by read
    :param N: (N>1) the order of the ngram e.g. N=2 gives a bigram
    :param k: optional add-k smoothing
    :returns: ngram and unigram
    """
    flat_corpus = [item for sublist in corpus for item in sublist]
    #ngram = defaultdict(Counter) #ngram[history][word] = #(history,word)
    # ngram = defaultdict(lambda: k/(N+kV), ngram)
    unigram = defaultdict(float, Counter(flat_data)) # default prob is 0.0           

    ## YOUR CODE HERE ##
    total_sum = sum(unigram.values())
    for w, v in unigram.items():
        unigram[w] = v/total_sum
    
    print("time to create the 1-gram: ", time.time() - start_time)
    #create the history for each n-gram made of n-1 words
    #history = Counter(tuple(window(flat_corpus, size=N-1)))
    print("time to create histories: ", time.time() - start_time)
    # nested dictionary to associate to each gram a word
    counter_model = defaultdict(lambda: defaultdict(int))
    probability_model = defaultdict(lambda: defaultdict(lambda: 0))
    # add padding at the beginnning and counting words for each history
    for sentence in corpus:
        for i in range(N-2):
            sentence.insert(0, "<s>")
            sentence.append("</s>")
        
        n_grams = tuple(window(sentence, size=N))
        
        for item in n_grams:
            history, word = tuple((item[:-1],item[-1]))
            counter_model[history][word] += 1
    #print(len(list(model.items())))
    #print(corpus[:2])
    print("time for couting words for each history: ", time.time() - start_time)
    # calculate probability with add-k method
    counter = 0
    for history in counter_model:
        total_count = float(sum(counter_model[history].values())) 
        default_value = ((counter_model[history][word] + k) / (total_count + k*len(unigram)))
        probability_model[history] = defaultdict(lambda: default_value, probability_model[history])
        for word in counter_model[history]:
            probability_model[history][word]
            if counter < 10:
                if counter == 1:
                    print(probability_model[history].values())
                print("model[history][word]", model[history][word])
                #print("total_count", total_count)
                print("default_value", default_value)
                counter += 1
        #model = defaultdict(lambda: defaultdict(lambda: 0)
        # ngram = defaultdict(lambda: k/(N+kV), ngram)
        #for word in model[history]:
        #    model[history][word] = (model[history][word] + k) / (total_count + k*len(unigram))
        #unseen_words = list(set(list(unigram.keys())) - set(list(model[history])))
        #for w in unseen_words:
            #model[history][word] = k / (total_count + k*len(unigram)) 
            
    
    print("time to train the N-gram: ", time.time() - start_time)
    return probability_model, unigram

#data, w2i, i2w = read(train_file)
#bigram, unigram = train_ngram(data, N=2, k=0)
#bigram_smoothed, unigram_smoothed = train_ngram(data, N=4, k=1)

def generate_sent(lm, N=2):
    ## YOUR CODE HERE ##
    
    sentence_finished = False
    sentence = ['<s>']*(N-1)
    
    while not sentence_finished:
        # keep on genereating words
        history = tuple(sentence[1-N:])
        # print(history)
        sample_word = sampleGram(lm, history)
        
        if sample_word is None: # draw random worn from list of unseen words
            sentence_finished = True
            
        sentence.append(sample_word)
        #print(sentence)
        if sentence[1-N:] == ["</s>"]*(N-1): # the sentence is finished
            sentence_finished = True
        
            
    return sentence
    
def sampleGram(model, history):
    u = random.random() # uniformly random number between 0 and 1
    p = 0
    
    for item in dict(model[history]).items():
        p += item[1]
        if p > u: 
            return item[0]


#data, w2i, i2w = read(train_file)
#bigram, unigram = train_ngram(data, N, k=0)
#bigram_smoothed, unigram_smoothed = train_ngram(data, N, k=1) 

#generate_sent(bigram, N=4)

[' '.join([str(elem) for elem in generate_sent(bigram, N=4)]) for i in range(5)]
