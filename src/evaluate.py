import math
import numpy as np

vsize = 100
vocab_file = 'vocab.dat'
weVec_file = 'v0_cbow_15/wordVector.dat'
question_file = 'questions-words.txt'
vfreqT = 5 

map_word2index = {}
map_index2word = {}

def buildVocab():
    ##build index and word table
    cc = 0
    print("build vocabulary...")
    with open(vocab_file, encoding='utf-8') as fh:
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
            if count < vfreqT: continue
            map_word2index[word] = cc
            map_index2word[cc] = word 
            cc += 1
    print("%d words" % (cc))
    ##build word vector from file
    print("construct word embedding matrix...")
    weVec = []
    with open(weVec_file) as fh:
        for wv in fh:
            wv = wv.strip()
            wvs = wv.split(',')
            tmp = [float(item) for item in wvs]
            weVec.append(tmp)
    weVec = np.mat(weVec) 

    print("calculat Euclidean distance for embedding matrix...")
    Ewhole = []
    m, n = weVec.shape
    for i in range(m): 
        row = weVec[i,:]
        Erow = math.sqrt(np.dot(row, row.T)[0,0])
        Ewhole.append(Erow)
    mEwhole = np.mat(Ewhole).reshape(m,1)
    
    return weVec, mEwhole
    
def cos_similarity(v0, v1):
    d0 = np.dot(v0, v0.T)[0,0]
    d1 = np.dot(v1, v1.T)[0,0]
    return np.dot(v0, v1.T)[0,0]/(math.sqrt(d0) * math.sqrt(d1))

def get_similarity(w0, w1, weVec):
    w00 = w0.lower()
    if w00 not in map_word2index: 
        print("not found word '%s' in vocabulary" % (w0))
        return 
    w10 = w1.lower()
    if w10 not in map_word2index: 
        print("not found word '%s' in vocabulary" % (w1))
        return 
    i0 = map_word2index[w00]
    i1 = map_word2index[w10]
    return cos_similarity(weVec[i0,:], weVec[i1,:])

def find_nearest(srcW, weVec, mEwhole, nearest=5):
    srcW0 = srcW.lower()
    if srcW0 not in map_word2index: 
        print("not found word '%s' in vocabulary" % (srcW))
        return 

    srcI = map_word2index[srcW0]

    srcWE = weVec[srcI,:]
    m,n = weVec.shape
    ##find nearest word simillar to srcWE
    sim_score = weVec * np.mat(srcWE.reshape(n,1))
    sim_score = sim_score / (mEwhole[srcI,0]*mEwhole)
    scoreI = np.argsort(sim_score, axis=0)
    result = []
    for i in range(2, nearest+2):
        result.append(map_index2word[scoreI[m-i,0]])
    return result 

##test case
weVec, mEwhole = buildVocab()


###nearest word
#print(get_similarity("day", "night", weVec))
print(find_nearest("good", weVec, mEwhole))
print(find_nearest("bad", weVec, mEwhole))
print(find_nearest("excellent", weVec, mEwhole))
print(find_nearest("awsome", weVec, mEwhole))
print(find_nearest("man", weVec, mEwhole))
print(find_nearest("woman", weVec, mEwhole))
print(find_nearest("teacher", weVec, mEwhole))
print(find_nearest("apple", weVec, mEwhole))
print(find_nearest("red", weVec, mEwhole))
exit(0)

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

##build the question dictionary
qDict = {}
dummy = 0
with open(question_file) as fh:
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
            if word not in map_word2index: break
            qiList.append(map_word2index[word])
        if len(qiList) != 4: 
            dummy += 1
            continue
        qList.append(qiList)
print("[checkerThread]: %d dummy questions" % (dummy))

qc = 0
tAccuracy = 0
for tName, tList in qDict.items():
    curAccuracy = 0
    print("[checkerThread]: %s test" % (tName))
    for [ta, tb, tc, td] in tList:
        curAccuracy += questionTest(ta, tb, tc, td, weVec, mEwhole) 
    subQc = len(tList)
    qc += subQc
    tAccuracy += curAccuracy
    print("[checkerThread]: %s accuracy: %f(%d/%d)" % (tName, curAccuracy/subQc, curAccuracy, subQc))
avgAcu = tAccuracy/qc
print("[checkerThread]: average accuracy: %f(%d/%d)" % (avgAcu, tAccuracy, qc))

print("[checkerThread]: done")   

