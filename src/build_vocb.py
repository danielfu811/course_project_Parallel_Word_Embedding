'''
    build vocabulary table from list of files.
'''
import re
import os
import nltk
import threading 
import queue

singleComma = set(["'s", "''", "'", "'ve", "'m", "'re", "'ll", "'d", "-", "."])

def file2tokens(file_name):
    with open(file_name) as fh:
        try:
            sample = fh.read()
        except: 
            print("can't process %s" % file_name)
            return None
    sample = sample.lower()
    sample = sample.replace('<br />', '')
    #sentences = nltk.sent_tokenize(sample)
    #tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    #tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    #chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    tokens = []
    #ne_list = []
    #for tree in chunked_sentences:
    #    for child in tree:
    #        if isinstance(child, nltk.tree.Tree):
    #            ne = '_'.join([leaf[0] for leaf in child.leaves()])
    #            tokens.append(ne)
    #            ne_list.append(ne)
    #        else:
    #            item = child[0]
    #            if item.isdigit(): tokens.append("@NUM")
    #            elif re.match(r'^\d*\.\d+([E,e]\d+)?$', item) != None: tokens.append("@FLOAT_NUM")
    #            else: tokens.append(item)
    #return tokens, ne_list 
    tokenized_sentences = nltk.word_tokenize(sample)
    for token in tokenized_sentences:
        item = token 
        if item in singleComma: 
            tokens.append(item)
            continue
        if item[0] == "'":
            tokens.append("'")
            item = item[1:]
        elif item[0] == '-': 
            if re.match(r'^-\d+$', item) != None: 
                tokens.append("@NEG_NUM") 
                continue
            tokens.append('-')
            item = item[1:]
        elif item[0] == '.': 
            i = 0
            while i < len(item) and item[i] == '.': i += 1
            tokens.append(item[0:i])
            if i == len(item): continue
            item = item[i:]
        #if not item.isalnum(): continue
        if item.isdigit(): tokens.append("@NUM")
        elif re.match(r'^\d{1,3}(,\d\d\d)+$', item) != None: tokens.append("@NUM")
        elif re.match(r'^\d+s$', item) != None: tokens.append("@NUMS") #70s, 60s
        elif re.match(r'^\d+-\d+$', item) != None: tokens.append("@RANGE") #70-60
        elif re.match(r'^\d*\.\d+([E,e]\d+)?$', item) != None: tokens.append("@FLOAT_NUM")
        elif re.match(r'^\d+/\d+$', item) != None: tokens.append("@RADIO")
        elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', item) != None: tokens.append("@DATE")
        else:
            if re.match(r'\d', item) != None: continue 
            tokens.append(item)
    return tokens

def gatherFile(wdir):
    '''
        gather all the files in 'wdir'.
    '''
    fileL = []
    curL = os.listdir(wdir)
    for item in curL:
        wpath = wdir + '/' + item
        if os.path.isfile(wpath): 
            fileL.append(wpath)
            continue
        if os.path.isdir(wpath):
            fileL.extend(gatherFile(wpath))
            continue
    return fileL

def worker(tname, wdir, resultQ):
    file_list = gatherFile(wdir)
    fileC = len(file_list)
    partVoc = {}
    print("%s: %d files to be processed" % (tname, fileC))
    i = 0
    for file_name in file_list:
        i += 1
        tokens = file2tokens(file_name)
        if tokens == None: continue
        for item in tokens:
            if item not in partVoc: partVoc[item] = 0
            partVoc[item] += 1

        if i % 100 == 0: 
            print("%s->%s: %d file left!!" % (tname, wdir, fileC-i))

    resultQ.put_nowait(partVoc)
    print("%s-> %s done" % (tname, wdir))

if __name__ == '__main__':
    vocab_file = "vocab.dat"

    dir_list = ['aclImdb/train/neg', 'aclImdb/train/pos', 'aclImdb/train/unsup',\
                'aclImdb/test/neg', 'aclImdb/test/pos']
    #dir_list = ['t0', 't1']

    wc = len(dir_list)
    resultQ = queue.Queue(wc) 
    threadL = []
    j = 0
    for wdir in dir_list:
        tname = "worker-%d" % (j)
        t = threading.Thread(target = worker, name = tname, args = (tname, wdir, resultQ))
        t.start()
        threadL.append(t)
        j += 1

    for j in range(wc):
        threadL[j].join()

    print("merge begin...")
    wordTotal = 0
    vocDict = resultQ.get_nowait().copy()
    while not resultQ.empty():
        partVoc = resultQ.get_nowait()
        for word, wc in partVoc.items():
            if word not in vocDict: vocDict[word] = 0
            vocDict[word] += wc
    
    print("%d different words in total" % len(vocDict))
    vocList = [(word, wc) for word, wc in vocDict.items()]
    vocList.sort(key = lambda item: item[1], reverse = True)
    print("write to file...")
    with open(vocab_file, "w") as fh:
        for item in vocList:
            fh.write("%s: %d\n" % (item[0], item[1]))
            wordTotal += item[1]
    print("total words: %d " % wordTotal)
    print("finish!!")
