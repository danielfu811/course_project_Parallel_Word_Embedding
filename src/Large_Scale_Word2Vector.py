'''
    large scale word2vector.
'''

import argparse
import sys
import tensorflow as tf
import numpy as np
import queue
import threading
import math
import os
import time
import random

###global Variables
##store the command line parameters
FLAGS = None
##vocabulary table: format (word: [index, frequency])
VOCAB = {}
totalWordCount = 0
sampleTh = float(0)
##exp table
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
expTable = np.zeros(EXP_TABLE_SIZE)
TAB_CONST = EXP_TABLE_SIZE / MAX_EXP / 2

##################################################################
def _debug_print_func(name, probe):
    print('{}: {}'.format(name, probe))
    return False

def buildSkipGram(wVector, sWeight, wInput, wOutput, wNeg, gPos, gNeg, width):
    ##collect the parameters
    mapInput = tf.gather(wVector, wInput)
    mapInput = tf.reshape(mapInput, [-1, width], name="mapInput")
    mapOutput = tf.gather(sWeight, wOutput, name="mapOutput")
    wNegFlat = tf.reshape(wNeg, [-1], name = "wNegFlat")
    mapNeg = tf.gather(sWeight, wNegFlat, name = "mapNeg")
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [mapInput.name, mapInput], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            mapInput = tf.identity(mapInput)
        debug_print_op = tf.py_func(_debug_print_func, [mapOutput.name, mapOutput], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            mapOutput = tf.identity(mapOutput)
        debug_print_op = tf.py_func(_debug_print_func, [mapNeg.name, mapNeg], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            mapNeg = tf.identity(mapNeg)
    ###################### partial dot multiplication #####################
    fPos = tf.matmul(mapInput, mapOutput, transpose_b=True, name="Pos") #shape [1, 2*win_size]
    tmp = tf.matmul(mapInput, mapNeg, transpose_b=True, name="Neg")
    fNeg = tf.reshape(tmp, [-1, FLAGS.neg_samples]) #shape [2*win_size, neg_samples]

    assert_inf_fPos = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(fPos))), \
                                [mapInput, mapOutput, mapNeg], name='assert_inf_fPos')
    assert_inf_fNeg = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(fNeg))), \
                                [mapInput, mapOutput, mapNeg], name='assert_inf_fNeg')
    assert_nan_fPos = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(fPos))), \
                                [mapInput, mapOutput, mapNeg], name='assert_nan_fPos')
    assert_nan_fNeg = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(fNeg))), \
                                [mapInput, mapOutput, mapNeg], name='assert_nan_fNeg')
    with tf.control_dependencies([assert_inf_fPos, assert_inf_fNeg, \
                                  assert_nan_fPos, assert_nan_fNeg]):
        fPos = tf.identity(fPos)
        fNeg = tf.identity(fNeg)
    ###################### partial gradient adjust ########################
    ####positive gradient: word vector
    deltaWV0 = tf.transpose(mapOutput)
    deltaWV1 = tf.multiply(deltaWV0, gPos)
    deltaWV2 = tf.transpose(deltaWV1)
    deltaWV3 = tf.reduce_sum(deltaWV2, axis=0)
    deltaWV4 = tf.reshape(deltaWV3, [-1, width], name="deltaWV4")
    if FLAGS.debug == 2:
        deltaWV4 = tf.Print(deltaWV4, [sWeight], message = "deltaWV4_input", summarize=100) 
        deltaWV4 = tf.Print(deltaWV4, [mapOutput, gPos, deltaWV4], message = "deltaWV4", summarize=100) 

    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaWV4.name, deltaWV4], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaWV4 = tf.identity(deltaWV4)
    ####positive gradient: softmax weight
    gPosShape = tf.shape(gPos)
    replica0 = tf.tile(mapInput, gPosShape)
    replica1 = tf.reshape(replica0, [-1, width])
    deltaSW0 = tf.transpose(replica1)
    deltaSW1 = tf.multiply(deltaSW0, gPos)
    deltaSW2 = tf.transpose(deltaSW1, name="deltaSW2")
    if FLAGS.debug == 2:
        deltaSW2 = tf.Print(deltaSW2, [deltaSW0, gPos, deltaSW2], message = "deltaSW2", summarize=100) 
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaSW2.name, deltaSW2], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaSW2 = tf.identity(deltaSW2)
    ####negtive gradient: word vector
    gNegFlat = tf.reshape(gNeg, [-1])
    mapNeg0  = tf.transpose(mapNeg)
    deltaWV5 = tf.multiply(mapNeg0, gNegFlat)
    deltaWV6 = tf.transpose(deltaWV5)
    deltaWV7 = tf.reduce_sum(deltaWV6, axis=0)
    deltaWV8 = tf.reshape(deltaWV7, [-1, width])
    if FLAGS.debug == 2:
        deltaWV8 = tf.Print(deltaWV8, [mapNeg, gNegFlat, deltaWV8], message = "deltaWV8", summarize=100) 
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaWV8.name, deltaWV8], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaWV8 = tf.identity(deltaWV8)
    ####negtive gradient: softmax weight
    gNegFlatS0 = tf.shape(gNegFlat)[0]
    replica2 = tf.tile(mapInput, [1,gNegFlatS0])
    replica3 = tf.reshape(replica2, [-1, width])
    replica4 = tf.transpose(replica3)
    deltaSW3 = tf.multiply(replica4, gNegFlat)
    deltaSW4 = tf.transpose(deltaSW3)
    if FLAGS.debug == 2:
        deltaSW4 = tf.Print(deltaSW4, [replica4, gNegFlat, deltaSW4], message = "deltaSW4", summarize=100) 
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaSW4.name, deltaSW4], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaSW4 = tf.identity(deltaSW4)
    #####adjust
    with tf.control_dependencies([deltaWV4, deltaWV8, deltaSW2, deltaSW4]):
        deltaWVA = deltaWV4 + deltaWV8
        newWV = tf.scatter_add(wVector, wInput, deltaWVA)
        newSWTmp = tf.scatter_add(sWeight, wOutput, deltaSW2)
        newSW = tf.scatter_add(newSWTmp, wNegFlat, deltaSW4)
        #mergeIndex = tf.concat(0, [wOutput, wNegFlat])
        #mergeDelta = tf.concat(0, [deltaSW2, deltaSW4])
        #newSW = tf.scatter_add(sWeight, mergeIndex, mergeDelta)
    if FLAGS.debug == 2:
        newWV = tf.Print(newWV, [wInput, wOutput, wNeg, gPos, gNeg], message = "input", summarize=100) 

    return (fPos, fNeg, newWV, newSW, wVector, sWeight)

def buildCBOW(wVector, sWeight, wInput, wOutput, wNeg, gPos, gNeg, width):
    ##collect the parameters
    tmpInput = tf.gather(wVector, wInput)
    hideAdd = tf.reduce_mean(tmpInput, axis=0)
    mapInput = tf.reshape(hideAdd, [-1, width], name="mapInput")
    tmpOutput = tf.gather(sWeight, wOutput)
    mapOutput = tf.reshape(tmpOutput, [-1, width], name="mapOutput")
    mapNeg = tf.gather(sWeight, wNeg, name="mapNeg")
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [mapInput.name, mapInput], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            mapInput = tf.identity(mapInput)
        debug_print_op = tf.py_func(_debug_print_func, [mapOutput.name, mapOutput], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            mapOutput = tf.identity(mapOutput)
        debug_print_op = tf.py_func(_debug_print_func, [mapNeg.name, mapNeg], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            mapNeg = tf.identity(mapNeg)
    ###################### partial dot multiplication #####################
    fPos = tf.matmul(mapInput, mapOutput, transpose_b=True) #shape [1,1]
    fNeg = tf.matmul(mapInput, mapNeg, transpose_b=True)#shape [1, neg_samples]

    assert_inf_fPos = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(fPos))), \
                                [wInput, wOutput, wNeg, mapInput, mapOutput, mapNeg], name='assert_inf_fPos')
    assert_inf_fNeg = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(fNeg))), \
                                [wInput, wOutput, wNeg, mapInput, mapOutput, mapNeg], name='assert_inf_fNeg')
    assert_nan_fPos = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(fPos))), \
                                [wInput, wOutput, wNeg, mapInput, mapOutput, mapNeg], name='assert_nan_fPos')
    assert_nan_fNeg = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(fNeg))), \
                                [wInput, wOutput, wNeg, mapInput, mapOutput, mapNeg], name='assert_nan_fNeg')
    with tf.control_dependencies([assert_inf_fPos, assert_inf_fNeg, \
                                  assert_nan_fPos, assert_nan_fNeg]):
        fPos = tf.identity(fPos)
        fNeg = tf.identity(fNeg)

    ###################### partial gradient adjust ########################
    ####positive gradient: word vector
    deltaWV0 = mapOutput * gPos[0,0]
    deltaWV1 = deltaWV0 / tf.to_float(tf.shape(wInput)[0])
    deltaWV2 = tf.tile(deltaWV1, [1, tf.shape(wInput)[0]])
    deltaWV3 = tf.reshape(deltaWV2, [-1, width], name="deltaWV3")
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaWV3.name, deltaWV3], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaWV3 = tf.identity(deltaWV3)
    ####positive gradient: softmax weight
    deltaSW0 = mapInput * gPos[0,0]
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, ["deltaSW0", deltaSW0], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaSW0 = tf.identity(deltaSW0)
    ####negtive gradient: word vector
    deltaWV4 = tf.multiply(tf.transpose(mapNeg), gNeg)
    deltaWV5 = tf.reduce_sum(tf.transpose(deltaWV4), axis=0)
    deltaWV6 = tf.reshape(deltaWV5, [-1]) / tf.to_float(tf.shape(wInput)[0])
    deltaWV7 = tf.tile(deltaWV6, tf.shape(wInput))
    deltaWV8 = tf.reshape(deltaWV7, [-1, width], name="deltaWV8")
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaWV8.name, deltaWV8], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaWV8 = tf.identity(deltaWV8)
    ####negtive gradient: softmax weight
    gNegFlat = tf.reshape(gNeg, [-1, FLAGS.neg_samples])
    replica0 = tf.tile(mapInput, tf.shape(gNegFlat))
    replica1 = tf.reshape(replica0, [-1, width])
    deltaSW1 = tf.multiply(tf.transpose(replica1), gNegFlat)
    deltaSW2 = tf.transpose(deltaSW1, name="deltaSW2")
    if FLAGS.debug == 1:
        debug_print_op = tf.py_func(_debug_print_func, [deltaSW2.name, deltaSW2], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            deltaSW2 = tf.identity(deltaSW2)
    #####adjust
    with tf.control_dependencies([deltaWV3, deltaWV8, deltaSW0, deltaSW2]):
        deltaWVA = deltaWV3 + deltaWV8
        newWV = tf.scatter_add(wVector, wInput, deltaWVA)
        newSWTmp = tf.scatter_add(sWeight, wOutput, deltaSW0)
        newSW = tf.scatter_add(newSWTmp, wNeg, deltaSW2)

    return (fPos, fNeg, newWV, newSW, wVector, sWeight)


def buildGraph(tskIndex, wInput, wOutput, wNeg, gPos, gNeg, vocabSize, ps_count):
    '''
    build the graph model: skip-gram and cbow, including two phace: partial dot multiplication and gradient adjust.
        tskIndex: the task index, between 0 ~ parameter_server_count -1.
        wInput: placeholder, index of input, according to the vocabulary table.
        wOutput: placeholder, index of output, according to the vocabulary table.
        wNeg:  placeholder, index of negtive samples, according to the vocabulary table.
        gPos: placeholder, positive gradient, for gradient adjust.
        gNeg: placeholder, negtive gradient, for gradient adjust.
        vocabSize: the size of vocabulary table.
        ps_count: the total count of PS server.
    the input only includes one window.
    input shape for skip-gram:
        wInput: [1]
        wOutput: [2*win_size]
        wNeg:   [2*win_size, neg_samples]
        gPos:   [1, 2*win_size]
        gNeg:   [2*win_size, neg_samples]
    input shape for cbow:
        wInput: [2*win_size]
        wOutput: [1]
        wNeg:   [neg_samples]
        gPos:   [1,1]
        gNeg:   [1,neg_samples]
    '''
    host = "/job:ps/task:%d" % (tskIndex)
    print("[buildGraph] host: %s" % (host))
    with tf.device(host):
        ##partial word vector and softmax weight
        width = int(FLAGS.vsize / ps_count)
        ####mean = 0, std = 1, word vector and soft max weight
        wVector = tf.Variable(tf.random_normal([vocabSize, width], dtype=tf.float32, seed=0), tf.float32, name='word-vector')
        sWeight = tf.Variable(tf.random_normal([vocabSize, width], dtype=tf.float32, seed=1), tf.float32, name='soft-weight')
        #wVector = tf.Variable(tf.ones([vocabSize, width], dtype=tf.float32), dtype=tf.float32, name='word-vector')
        #sWeight = tf.Variable(tf.ones([vocabSize, width], dtype=tf.float32), dtype=tf.float32, name='soft-weight')
        ###build model
        if FLAGS.model == 'skip-gram':
            ##skip-gram model
            return buildSkipGram(wVector, sWeight, wInput, wOutput, wNeg, gPos, gNeg, width)
        else:
            ##cbow model
            return buildCBOW(wVector, sWeight, wInput, wOutput, wNeg, gPos, gNeg, width)

##################################################################
def buildVocab():
    '''
        build vocabulary table, return the table size.
        the vocabulary file should have the following format:
            word0: frequency
            word1: frequency
    '''
    cc = 0
    global totalWordCount
    global sampleTh
    with open(FLAGS.vocab, encoding='utf-8') as fh:
        for item in fh:
            item = item.strip()
            if len(item) == 0: continue
            #word, freq = item.split(':')
            i = len(item)-1
            while i > 0: 
                if item[i] == ':': break
                i -= 1
            word = item[0:i]
            freq = item[i+1:]
            word = word.strip()
            freq = freq.strip()
            count = int(freq)
            if count < FLAGS.vfreqT: continue
            totalWordCount += count
            VOCAB[word] = [cc, count]
            cc += 1
            #print("%d: %s" % (cc, word))
    sampleTh = cc * FLAGS.sample
    return cc 

def buildExpTab():
    '''
        build exp table. f(x) = exp(x) / (1 + exp(x))
    '''
    for i in range(EXP_TABLE_SIZE):
        expTable[i] = math.exp((i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        expTable[i] = expTable[i] / (expTable[i] + 1)

def getSigmod(x):
    '''
        look up the exp table, return exp(x)/ (1 + exp(x)).
    '''
    if x > MAX_EXP: g = float(1.0)
    elif x < -MAX_EXP: g = float(0.0)
    else:
        index = int((x + MAX_EXP) * TAB_CONST)
        if index >= EXP_TABLE_SIZE: index = EXP_TABLE_SIZE - 1
        g = expTable[index]
    return g

def getNextSeed(seed):
    nextSeed = (seed * 25214903917 + 11) & 0xffffffffffffffff
    return nextSeed

def getRandomIndex(seed, limit):
    seed = getNextSeed(seed)
    target = (seed >> 16) % limit
    return seed, target

def generateRandIndexList(seed, limit, n, exclude):
    result = []
    i = 0
    while i < n:
        seed, index = getRandomIndex(seed, limit)
        if index == exclude: continue
        if index in result: continue
        result.append(index)
        i += 1
    return seed, result

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

def wordSample(token, seed):
    global sampleTh
    rate = sampleTh / VOCAB[token][1]
    ran = (math.sqrt(1/rate) + 1) * rate
    nseed = getNextSeed(seed)
    if ran < (nseed & 0xffff)/65536.0: return True, nseed 
    return False, nseed 

def token2index(tokens, seed):
    tiList = []
    nseed = seed
    curTC = 0
    for token in tokens:
        if token in VOCAB: 
            ##sample words, remove frequent words
            curTC += 1
            status, nseed = wordSample(token, nseed)
            if status == True: continue
            tiList.append(VOCAB[token][0])
    return tiList, nseed, curTC

def trainPatternGenerator(srcF, seed):
    '''
        generate train patter in file 'srcF'.
        output for skip-gram: win(1), wout(2*win_size), wneg(2*win_size, neg_samples)
        output for cbow: win(2*win_size), wout(1), wneg(neg_samples)
    '''
    import build_vocb
    ##tokennize
    tokens = build_vocb.file2tokens(srcF)
    if tokens == None: 
        yield None, None, None, seed, 0
        return 
    ##token to index   
    tokenIList, nseed, curTC = token2index(tokens, seed)
    #print(tokenIList)
    ##generate train pattern
    tc = len(tokenIList)
    vocabC = len(VOCAB)
    for ci in range(tc):
        t_in = []
        t_out = []
        t_neg = []
        #print("seed: {}".format(seed))
        #print("nseed: {}".format(nseed))
        ##dynamic window size
        nseed = getNextSeed(nseed)
        winOffset = nseed % FLAGS.win_size 
        if FLAGS.model == "skip-gram":
            ####input
            t_in.append(tokenIList[ci])
            ####output
            j = ci - FLAGS.win_size + winOffset
            while j != ci: 
                if j >= 0: t_out.append(tokenIList[j])
                j += 1
            j = ci + 1
            wt = ci + FLAGS.win_size - winOffset
            while j <= wt and j < tc: 
                t_out.append(tokenIList[j])
                j += 1
            ####negtive sample
            outC = len(t_out)
            for j in range(outC): 
                nseed, negL = generateRandIndexList(nseed, vocabC, FLAGS.neg_samples, t_out[j]) 
                t_neg.append(negL)
        else:  #cbow model
            ####input
            j = ci - FLAGS.win_size + winOffset
            while j != ci: 
                if j >= 0: t_in.append(tokenIList[j])
                j += 1
            j = ci + 1
            wt = ci + FLAGS.win_size - winOffset
            while j <= wt and j < tc: 
                t_in.append(tokenIList[j])
                j += 1
            ####output
            t_out.append(tokenIList[ci])
            ####negtive sample
            nseed, t_neg = generateRandIndexList(nseed, vocabC, FLAGS.neg_samples, t_out[0]) 

        yield t_in, t_out, t_neg, nseed, curTC

def combineWordVector(sessL, graphL, vocabSize, genMatrix=False):
    '''
        combine the word vector.
    '''
    ## get partial word vector from PS servers.
    print("[combineWordVector] fetch word vector from PS server...")
    wvL = []
    ps_count = len(sessL)
    for i in range(ps_count):
        wvL.append(sessL[i].run(graphL[i][-2]))
    assert len(wvL) == ps_count, \
           "the gather list size should be %d, but %d" % (ps_count, len(wvL))
    ##combine
    print("[combineWordVector] begin to combine")
    if genMatrix == True:
        wordVs = np.zeros((vocabSize, FLAGS.vsize), dtype=np.float32)
        psW = int(FLAGS.vsize/ps_count)
        start = 0
        for i in range(ps_count):
            wordVs [:,start:start+psW] = wvL[i]
            start += psW
        return np.mat(wordVs)

    with open(FLAGS.output, "w") as fh:
        for i in range(vocabSize):
            wv = ""
            j = 0
            for ps in wvL:
                partV = ps[i].tolist()
                wv += ','.join(str(e) for e in partV) 
                j += 1
                if j < ps_count: wv += ','
            ##write to file
            fh.write("%s\n" % wv)

##################################################################
###checker thread to calculate the quality of word vectors
class Cmd:
    def __init__(self, wid, cmd):
        self.wid = wid #worker id
        self.cmd = cmd #string, 'continue' or 'finish'

def cos_similarity(v0, v1):
    d0 = np.dot(v0, v0.T)[0,0]
    d1 = np.dot(v1, v1.T)[0,0]
    return np.dot(v0, v1.T)[0,0]/(math.sqrt(d0) * math.sqrt(d1))

def calEuclidean(weVec): 
    Ewhole = []
    m, n = weVec.shape
    for i in range(m): 
        row = weVec[i,:]
        Erow = math.sqrt(np.dot(row, row.T)[0,0])
        Ewhole.append(Erow)
    mEwhole = np.mat(Ewhole).reshape(m,1)
    return mEwhole

def questionTest(ta, tb, tc, td, weVec, mEwhole):
    t0 = weVec[tb,:] - weVec[ta,:]
    genWE = weVec[tc,:] + t0
    m, n = weVec.shape
    ##find a word simillar to genWE
    #print(genWE.shape)
    sim_score = np.mat(weVec) * np.mat(genWE.reshape(n,1))
    sim_score = sim_score /(math.sqrt(np.dot(genWE, genWE.T)[0,0]) * mEwhole)
    scoreI = np.argsort(sim_score, axis=0)
    candidates = scoreI[m-5:m-1,:]
    #print(candidates)
    if td in candidates: return 1
    return 0

def checkerThread(sessL, graphL, vocabSize, workersC, w2cQueue, c2wQueue):
    '''
    every iteration is done, the checkerThread is waken up, and make question test.
    if the question test reaches a threshold, it would send finished training command.
    sessL: session list.
    graphL: graph list.
    vocabSize: size of vocabulary.
    workersC: the count of total workers.
    w2cQueue: worker to checker queue, the element is the worker id.
    c2wQueue: checker to worker, command element as class Cmd.
    '''
    ##build the question dictionary
    qDict = {}
    dummy = 0
    with open(FLAGS.question_file) as fh:
        for line in fh:
            if line[0] == ':':
                qName = line[1:].strip()
                qList = []
                qDict[qName] = qList
                continue
            line = line.strip() 
            items = line.lower().split(' ') 
            ##to index
            qiList = []
            for word in items:
                if word not in VOCAB: break
                qiList.append(VOCAB[word][0])
            if len(qiList) != 4: 
                dummy += 1
                continue
            qList.append(qiList)
    print("[checkerThread]: %d dummy questions" % (dummy))
    for tName, tList in qDict.items():
        print("[checkerThread]: %s-> %d questions" % (tName, len(tList)))

    waitList = list(range(workersC))
    for i in range(FLAGS.iteration):
        print("[checkerThread]: wait for workers...")
        while len(waitList) > 0:
            wid = w2cQueue.get()
            waitList.remove(wid)
        print("[checkerThread]: %d iteration done" % (i))
        print("[checkerThread]: fetch word embeding vectors")
        weVec = combineWordVector(sessL, graphL, vocabSize, genMatrix=True)
        mEwhole = calEuclidean(weVec)
        qc = 0
        tAccuracy = 0
        for tName, tList in qDict.items():
            curAccuracy = 0
            #print("[checkerThread]: %s test" % (tName))
            for [ta, tb, tc, td] in tList:
                curAccuracy += questionTest(ta, tb, tc, td, weVec, mEwhole) 
            subQc = len(tList)
            qc += subQc
            tAccuracy += curAccuracy
            print("[checkerThread]: %s accuracy: %f(%d/%d)" % (tName, curAccuracy/subQc, curAccuracy, subQc))
        avgAcu = tAccuracy/qc
        print("[checkerThread]: %d iteration average accuracy: %f(%d/%d)" % (i, avgAcu, tAccuracy, qc))
        if i == FLAGS.iteration-1: break
        if avgAcu > FLAGS.embeding_quality:
            print("[checkerThread]: %d iteration reach threshold: %f" % (i, FLAGS.embeding_quality))
            for j in range(workersC):
                c2wQueue.put_nowait(Cmd(j, "finish"))
                print("[checkerThread]: sending worker_%d finish" % (j))
            break
        else:
            ###training continue
            waitList = list(range(workersC))
            for j in range(workersC):
                c2wQueue.put_nowait(Cmd(j, "continue"))
                print("[checkerThread]: sending worker_%d continue" % (j))

    print("[checkerThread]: done")   

##################################################################
###worker function 
curTrainWords = int(0)

lock = threading.Lock()

def setWordCounter(trainedW):
    global curTrainWords
    with lock: curTrainWords += trainedW

def checkGraph(stage, sessL, graphL):
    cc = len(sessL)
    chkSumWV = 0
    chkSumSW = 0
    for i in range(cc):
        fp, fn, adjWV, adjSW, wv, sw = graphL[i] 
        curWV, curSW = sessL[i].run([wv, sw])
        if i == 0: 
            m, n = curWV.shape
            chkSumWV = m*n
            m, n = curSW.shape
            chkSumSW = m*n
            bakWV = curWV 
            bakSW = curSW
            continue
        if np.sum(curWV == bakWV) != chkSumWV or np.sum(curSW == bakSW) != chkSumSW:
            print("%d stage: %d" % (stage, i-1))
            print(bakWV)
            print(bakSW)
            print("####################################")
            print("%d stage: %d" % (stage, i))
            print(curWV)
            print(curSW)
            exit(1)
        bakWV = curWV 
        bakSW = curSW       

def dotMultiplyThread(tname, sess, graph, dotPL, resultQ):
    fp, fn, adjWV, adjSW, wv, sw = graph 
    FP, FN = sess.run([fp, fn], dotPL)
    resultQ.put_nowait((FP, FN))
    #print("{}: {}, {}".format(tname, FP, FN))
    return

def adjustThread(tname, sess, graph, adjPL):
    fp, fn, adjWV, adjSW, wv, sw = graph 
    sess.run([adjWV.op, adjSW.op], adjPL)
    #print("wv-{}: {}".format(tname, sess.run(wv)))
    #print("sw-{}: {}".format(tname, sess.run(sw)))
    return

def mergeDot(resultQ, alpha):
    ##merge 
    FP, FN = resultQ.get_nowait()
    mFP = np.zeros(FP.shape)
    mFN = np.zeros(FN.shape)
    mFP += FP
    mFN += FN
    while not resultQ.empty():
        FP, FN = resultQ.get_nowait()
        mFP += FP
        mFN += FN
    ##calculate gradient 
    m, n = mFP.shape
    for i in range(m):
        for j in range(n): 
            mFP[i,j] = getSigmod(mFP[i,j])
    m, n = mFN.shape
    for i in range(m):
        for j in range(n): 
            mFN[i,j] = getSigmod(mFN[i,j])
    mFP = alpha * (1 - mFP)
    mFN = -alpha * mFN

    return mFP, mFN

def workerThread(tname, wid, sessL, graphL, pdict, wdir, w2cQueue, c2wQueue):
    ##get the whole file list in the 'wdir'
    fileL = gatherFile(wdir) 
    print("%s: %d file in total!!" % (tname, len(fileL)))
    ##training 
    curCount = 0
    lastCount = 0
    ps_count = len(sessL)
    alpha = FLAGS.alpha
    alplaT = FLAGS.alpha * 0.0001
    resultQ = queue.Queue(ps_count) ##store the dotMultiply from PS server
    psTL = [] ##store thead handle  
    seed = wid
    fileC = 0
    for i in range(FLAGS.iteration):
        print("%s: iteration %d" % (tname, i))
        random.shuffle(fileL)
        for srcF in fileL:
            ##generate training input from file.
            curTC = 0
            for t_in, t_out, t_neg, nseed, curTC in trainPatternGenerator(srcF, seed):
                if t_in == None: continue 
                if len(t_in) == 0 or len(t_out) == 0: continue
                #print("##################################")
                #print("{}: {}".format(tname, t_in))
                #print("{}: {}".format(tname, t_out))
                #print("{}: {}".format(tname, t_neg))
                #print(nseed)
                #generate dotMultiply placeholder 
                dotPL = {pdict["in"]: t_in, pdict["out"]: t_out, pdict["neg"]: t_neg} 
                #dotMultiply 
                for j in range(ps_count):
                    tDname = "%s_dot_%d" % (tname, j)
                    t = threading.Thread(target = dotMultiplyThread, name = tDname,\
                                         args = (tDname, sessL[j], graphL[j], dotPL, resultQ))
                    t.start()
                    psTL.append(t)
                ###wait
                for psT in psTL: psT.join()
                psTL.clear()
                ##merge result
                assert resultQ.qsize() == ps_count, \
                       "the dot queue size should be %d, but %d" % (ps_count, resultQ.qsize())
                gPos, gNeg = mergeDot(resultQ, alpha)
                #print("{}:{}".format(tname, gPos))
                #print("{}:{}".format(tname, gNeg))
                #generate dotMultiply placeholder 
                adjPL = {pdict["in"]: t_in, pdict["out"]: t_out, pdict["neg"]: t_neg,\
                         pdict["gPos"]: gPos, pdict["gNeg"]: gNeg}
                ##adjust
                for j in range(ps_count):
                    tAname = "%s_adj_%d" % (tname, j)
                    t = threading.Thread(target = adjustThread, name = tAname,\
                                         args = (tAname, sessL[j], graphL[j], adjPL))
                    t.start()
                    psTL.append(t)
                ###wait
                for psT in psTL: psT.join()
                psTL.clear()
                seed = nseed
                #checkGraph(curCount, sessL, graphL)

            ### one file finish, tune the learning rate every 10000 
            curCount += curTC 
            #if (curCount > 100): exit(1)
            lastFinishW = curCount - lastCount
            if lastFinishW >= 10000: 
                setWordCounter(lastFinishW)
                lastCount = curCount
                curPro = curTrainWords / (FLAGS.iteration * totalWordCount)
                alpha = FLAGS.alpha * (1 - curPro)
                if alpha < alplaT: alpha = alplaT 
                print("[{}] iteration {}: alpha {}, {}/{}%!".format(tname, i, alpha, curTrainWords, int(curPro*100)))
            fileC += 1
            if fileC % 30 == 0: 
                print("[%s] iteration %d: %d files done!" % (tname, i, fileC))
        
        print("[%s] iteration %d is done!!" % (tname, i))
        print("[%s] send wid %d to checker" % (tname, wid))
        w2cQueue.put_nowait(wid)
        if i == FLAGS.iteration-1: break
        print("[%s] wait for checker command" % (tname))
        while True:
            cmd = c2wQueue.get()
            if cmd.wid != wid: 
                c2wQueue.put_nowait(cmd)
                time.sleep(wid+1)
                continue
            break
        print("[%s] receive %s command" % (tname, cmd.cmd))
        if cmd.cmd == 'finish': break

    print("%s done, %d file in total!!" % (tname, len(fileL)))

##################################################################
def test_skip_gram(ps_count):
    vocabSize = 10
    ##build graph model
    t_in = tf.placeholder(tf.int32)
    t_out = tf.placeholder(tf.int32)
    t_neg = tf.placeholder(tf.int32)
    t_gPos = tf.placeholder(tf.float32)
    t_gNeg = tf.placeholder(tf.float32)
    (fp, fn, adjWV, adjSW, wv, sw) = buildGraph(0, t_in, t_out, t_neg, t_gPos, t_gNeg, vocabSize, ps_count)
    ##setup sessions
    sess = tf.Session()
    ##init the word vector and weight
    print("init....")
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(wv))
    print("dot multiplication...")
    tmpFP, tmpFN = sess.run([fp, fn], {t_in:[0], t_out:[2, 3], t_neg:[[5,6], [7,8]]})
    print(tmpFP)
    print(tmpFN)
    print("gradient adjust...")
    sess.run([adjWV.op, adjSW.op], {t_in:[0], t_out:[2, 3], t_neg:[[5,6],[7,8]], t_gPos:tmpFP, t_gNeg:tmpFN})
    print(sess.run(wv))
    print(sess.run(sw))
    print("#####################################################")
    print("dot multiplication...")
    tmpFP, tmpFN = sess.run([fp, fn], {t_in:[1], t_out:[2, 3], t_neg:[[5,6], [7,8]]})
    print(tmpFP)
    print(tmpFN)
    print("gradient adjust...")
    sess.run([adjWV.op, adjSW.op], {t_in:[1], t_out:[2, 3], t_neg:[[5,6],[7,8]], t_gPos:tmpFP, t_gNeg:tmpFN})
    print(sess.run(wv))
    print(sess.run(sw))
    print("finish!!")

def test_cbow(ps_count):
    vocabSize = 10
    ##build graph model
    t_in = tf.placeholder(tf.int32, [None])
    t_out = tf.placeholder(tf.int32)
    t_neg = tf.placeholder(tf.int32)
    t_gPos = tf.placeholder(tf.float32)
    t_gNeg = tf.placeholder(tf.float32)
    (fp, fn, adjWV, adjSW, wv, sw) = buildGraph(0, t_in, t_out, t_neg, t_gPos, t_gNeg, vocabSize, ps_count)
    ##setup sessions
    sess = tf.Session()
    ##init the word vector and weight
    print("init....")
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(wv))
    print("dot multiplication...")
    tmpFP, tmpFN = sess.run([fp, fn], {t_in:[0, 1], t_out:[2], t_neg:[5,6]})
    print(tmpFP)
    print(tmpFN)
    print("gradient adjust...")
    sess.run([adjWV.op, adjSW.op], {t_in:[0, 1], t_out:[2], t_neg:[5,6], t_gPos:tmpFP, t_gNeg:tmpFN})
    print(sess.run(wv))
    print(sess.run(sw))
    print("#####################################################")
    print("dot multiplication...")
    tmpFP, tmpFN = sess.run([fp, fn], {t_in:[1,2], t_out:[3], t_neg:[7,8]})
    print(tmpFP)
    print(tmpFN)
    print("gradient adjust...")
    sess.run([adjWV.op, adjSW.op], {t_in:[1, 2], t_out:[3], t_neg:[7,8], t_gPos:tmpFP, t_gNeg:tmpFN})
    print(sess.run(wv))
    print(sess.run(sw))
    print("finish!!")

def testExpTable():
    buildExpTab()
    print(getSigmod(0))
    print(getSigmod(MAX_EXP))
    print(getSigmod(-MAX_EXP))
    print(getSigmod(10))
    print(getSigmod(-10))
    print(getSigmod(1))
    print(getSigmod(-1))
    print("finish!!")
    exit(0)

def testRandomIndex():
    seed = 0
    print(generateRandIndexList(seed, 100, 10, 0))
    print("finish!!")
    exit(0)

def testGatherFile():
    print(gatherFile('./'))
    print("finish!!")
    exit(0)

def testBuildVocab():
    print(buildVocab())
    print(totalWordCount)
    print(VOCAB)
    print("#################################")
    srcF = "test/0_3.txt"
    seed = 0 
    for t_in, t_out, t_neg, nseed, curTC in trainPatternGenerator(srcF, seed):
        if t_in == None: continue 
        print("#################################")       
        print("input: {}".format(t_in))
        print("output: {}".format(t_out))
        print("neg sample: {}".format(t_neg))
        print("test seed: {}".format(nseed))
        seed = nseed
    print("finish!!")
    exit(0)

def testChecker(sessL, graphL, vocabSize):
    workersC = 3
    w2cQueue = queue.Queue(workersC)
    c2wQueue = queue.Queue(workersC)
    ##start checker
    print("[testChecker] start checkerThread...")
    checker = threading.Thread(target = checkerThread, name = "checker", \
                               args = (sessL, graphL, vocabSize, workersC, w2cQueue, c2wQueue))
    checker.start()

    ##test 
    time.sleep(3)
    for j in range(FLAGS.iteration):
        for i in range(workersC): w2cQueue.put_nowait(i)
        if j == FLAGS.iteration-1: break
        time.sleep(3)
        flag = 0
        for i in range(workersC):
            cmd = c2wQueue.get()
            print("[testChecker] get command: %d->%s" % (cmd.wid, cmd.cmd))
            if cmd.cmd == 'finish': flag += 1
        if flag == 3: break
        time.sleep(10)

    ##combine word vector
    checker.join()
    print("[testChecker: finish!")
    exit(0)


##################################################################################################################
def main(_):
    # parameter server list.
    ps_hosts = FLAGS.ps_hosts.split(",")
    ##get the count of PS server, and this is also the way to split word vector.
    ##each PS server hold two vsize/ps_hosts * table_size, one for word vector, one for weight.
    ps_count = len(ps_hosts)
    print("[main] count of PS server: %d" % ps_count)

    # Create a cluster from the parameter server.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts})

    if FLAGS.job_name == "ps":
        # Create and start a server for the local task.
        print("[main] PS server index: %d" % FLAGS.task_index)
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        server.join()
    elif FLAGS.job_name == "master":
        #test_skip_gram(ps_count)
        #test_cbow(ps_count)
        #testExpTable()
        #testRandomIndex()
        #testGatherFile()
        #testBuildVocab()
        ############################################################################################
        ##build vocabulary table
        print("[main] create vocabulary table...")
        vocabSize = buildVocab()
        print("[main] vocabSize: %d" % vocabSize)
        print("[main] total train words: %d" % totalWordCount)

        ##build exp table
        print("[main] build exp table...")
        buildExpTab()

        ##build graph model
        print("[main] build graph ...")
        graphL = []
        p_in = tf.placeholder(tf.int32, [None])
        p_out = tf.placeholder(tf.int32, [None])
        p_neg = tf.placeholder(tf.int32)
        p_gPos = tf.placeholder(tf.float32)
        p_gNeg = tf.placeholder(tf.float32)
        pdict = dict(zip(["in", "out", "neg", "gPos", "gNeg"], [p_in, p_out, p_neg, p_gPos, p_gNeg]))
        for i in range(ps_count):
            graphL.append(buildGraph(i, p_in, p_out, p_neg, p_gPos, p_gNeg, vocabSize, ps_count))

        ##setup sessions
        sessL = []
        for i in range(ps_count):
            node = "grpc://" + ps_hosts[i]
            sessL.append(tf.Session(node))

        #init the word vector and weight
        print("[main] init....")
        init = tf.global_variables_initializer()
        sessL[0].run(init)
        
        #testChecker(sessL, graphL, vocabSize)
        #for graph visualing 
        #train_writer = tf.summary.FileWriter('./log')
        #train_writer.add_graph(sessL[0].graph)
        #train_writer.close() 
        #exit(0)
    
        ##setup worker thread
        print("[main] setup worker...")
        workerL = []  # store the worker thread handle
        dirL = FLAGS.work_folder.split(',')
        print("[main] %d worker threads" % len(dirL))
        i = 0
        workersC = len(dirL)
        w2cQueue = queue.Queue(workersC)
        c2wQueue = queue.Queue(workersC)
        for workDir in dirL:
            tname = "worker_%d" % (i)
            print("[main] start %s ..." % tname)
            t = threading.Thread(target = workerThread, name = tname, \
                                 args = (tname, i, sessL, graphL, pdict, workDir, w2cQueue, c2wQueue))
            t.start()
            workerL.append(t)
            i += 1

        ##start checker
        print("[main] start checkerThread...")
        checker = threading.Thread(target = checkerThread, name = "checker", \
                                   args = (sessL, graphL, vocabSize, workersC, w2cQueue, c2wQueue))
        checker.start()
        ##combine word vector
        print("[main] wait for all workers...")
        for workerT in workerL: workerT.join()
        print("[main] all workers are done...")
        print("[main] wait for checker...")
        checker.join()
        print("[main] checker done...")
        print("[main] begin to combine...")
        combineWordVector(sessL, graphL, vocabSize)
        print("[main] finish!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'master'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    # Flags for defining the word vector parameters
    parser.add_argument(
        "--win_size",
        type=int,
        default=5,
        help="the size of training window"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="skip-gram",
        help="'skip-gram'(default) or 'cbow'"
    )
    parser.add_argument(
        "--neg_samples",
        type=int,
        default=10,
        help="the negtive sample count for negtive sampling"
    )
    parser.add_argument(
        "--vsize",
        type=int,
        default=100,
        help="the dimension size of word vector, default is 100"
    )
    # Flags for defining training parameters
    parser.add_argument(
        "--iteration",
        type=int,
        default=15,
        help="the iteration count, default is 5"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="the start learning rate"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.001,
        help="word sample rate, to recude frequent words"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="vocab.dat",
        help="the path to the vocabulary table"
    )
    parser.add_argument(
        "--vfreqT",
        type=int,
        default=5,
        help="the word frequency threshold to generate vocabulary"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wordVector.dat",
        help="the output word vector file name, default is 'wordVector.dat'"
    )
    parser.add_argument(
        "--work_folder",
        type=str,
        default="",
        help="Comma-separated list of work folder paths, which store the text file"
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default="questions-words.txt",
        help="questions, words file for checker"
    )
    parser.add_argument(
        "--embeding_quality",
        type=float,
        default=0.8,
        help="average cos similality in question_file"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level, default is 0"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("################## parameters ####################")
    print(FLAGS)
    print("##################################################")
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
