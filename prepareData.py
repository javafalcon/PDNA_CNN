# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:57:47 2019

@author: Administrator
"""
import numpy as np
from tools import aa2code
import re
from sklearn.utils import shuffle

# Use digital codeing for aa 
# Ref: Xiao,X. et al, 2004, Digital coding for amino acid based on cellular automata
XiaoInfoCode={'A':'11001','C':'01111','D':'11100','E':'11101',
          'F':'01011','G':'11110','H':'00101','I':'10010',
          'K':'10100','L':'00011','M':'10011','N':'10101',
          'P':'00001','Q':'00100','R':'00110','S':'01001',
          'T':'10000','V':'11010','W':'01110','Y':'01100',
          'X':'00000'}
# 处理PDNA-62数据集
def readPDNA62():
    pdna_seqs_62 = {}
    pdna_sites_62 = {}
    with open(r'PDNA_Data\Supplementary_data_S3_010913.doc','r',encoding='ISO-8859-1') as fr:
        key = ""
        start = False
        for line in fr:
            line = line.strip("\n")
            if line.startswith('.end'):
                break
            if line.startswith(r'.\lab'):
                start = True
                ls = line.split(" ")
                label = ls[0].split("\\")
                label = label[2]
                k = label.rindex(".",0,-1) 
                key = label[:k]
                print(key)
                pdna_seqs_62[key] = []
                pdna_sites_62[key] = []
            elif start:
                ls = line.split(" ")
                ind = 1
                
                for s in ls[1:]:
                    if s:
                        if ind == 1:
                            nsite = int(s)
                        elif ind == 2:
                            aa = s
                        elif ind == 4:
                            sign = s
                            break
                        ind += 1
                if sign == '1':
                   pdna_sites_62[key].append(nsite)
                pdna_seqs_62[key].append(aa) 
    return pdna_seqs_62, pdna_sites_62

def readPDNA224():
    pdna_seqs_224 = {}
    pdna_sites_224 = {}
    with open(r'PDNA_Data\Supplementary_data_S1_010913.doc','r',encoding='ISO-8859-1') as fr:
        sequencesFlag, sitesFlag = False, False
        for line in fr:
            line = line.replace("\n","")
            line = line.replace("\x0c","")
            if 'The sequences of PDNA-224' in line:
                sequencesFlag = True
                continue
            elif 'The binding sites information' in line:
                sequencesFlag, sitesFlag =False, True
                continue
            if sequencesFlag:
                if len(line.strip()) == 0:
                    continue
                if line.startswith('>'):
                    key = line[1:]
                else:
                    pdna_seqs_224[key] = pdna_seqs_224.get(key,'') + line
                
            elif sitesFlag:
                if len(line.strip()) == 0:
                    continue
                if line.startswith('>'):
                    key = line[1:]
                    pdna_sites_224[key] = []
                else:
                    line = line.rstrip("\t")
                    for s in line.split("\t"):
                        try:
                            pdna_sites_224[key].append(int(s))
                        except ValueError:
                            break
    return pdna_seqs_224, pdna_sites_224
              
def getTrainingDataset(pseqs:dict, psites:dict, windown_wise:int):
    keys = pseqs.keys()
    posseqs, negseqs = [], []
    for key in keys:
        seq = pseqs[key]
        n = len(seq)
        #给序列左右各填充windown_wise个'X'
        seq = list(seq)
        for k in range(11):
            seq.insert(0,'#')
            seq.append('#')
        seq = "".join(seq)    
        site = psites[key]
        
        #由于左边前windown_wise是填空的字符，所以从真实的序列从windown_wise开始
        #右边后windown_wise是填充字符，所以真实序列到n+windown_wise-1结束
        for i in range(windown_wise, n+windown_wise):
            start = i- windown_wise
            end = i + windown_wise+1
            seqseg = seq[start:end]
            if i-windown_wise+1 in site:
                posseqs.append(seqseg)
            else:
                negseqs.append(seqseg)
    return posseqs, negseqs


def naChaosGraph_codes(naseq, width=32, hight=32):
    g = np.zeros(shape=(width, hight))
    i,j = width//2, hight//2
    for c in naseq:
        if c == 'U':
            x,y = 0,0
        elif c == 'C':
            x,y = width,0
        elif c == 'A':
            x,y = width, hight
        elif c == 'G':
            x,y = 0,hight
            
        tx, ty = (i + x)//2, (j + y)//2
        g[tx,ty] = g[tx,ty] + 1
        i,j = tx,ty
    return g

def naChaosGraphVector(naseq, width=1., hight=1.):
    n = len(naseq)
    g = np.zeros(shape=(n,2))
    
    i,j = width/2, hight/2
    for k in range(n):
        c = naseq[k]
        if c == 'U':
            x,y = 0,0
        elif c == 'C':
            x,y = width,0
        elif c == 'A':
            x,y = width, hight
        elif c == 'G':
            x,y = 0,hight
        else:
            x,y = width/2, hight/2
        tx, ty = (i + x)/2, (j + y)/2
        g[k][0], g[k][1] = tx, ty
    return g

def aaseq2naseq(seq):
    seq = seq.upper()
    seq = re.sub('[ZUB]','',seq)
    seq = seq.strip()
    naseq = ""
    for c in seq:
        naseq += aa2code(c) 
    return naseq

def protXiaoInfoCode(seq):
    m = []
    for aa in seq:
        al = []
        for c in XiaoInfoCode[aa]:
            al.append(eval(c))
        m.append(al)
    return np.array(m)


def genEnlargedData(pseqs, nseqs, num_enlarg=0):
    X,y = [],[]
    if num_enlarg == 0:
        num_enlarg = len(nseqs)//len(pseqs)
    for seq in pseqs:
        for k in range(num_enlarg):
            naseq = aaseq2naseq(seq)
            X.append(naChaosGraphVector(naseq))
            y.append([0.,1.])
    for seq in nseqs:
        naseq = aaseq2naseq(seq)
        X.append(naChaosGraphVector(naseq))
        y.append([1.,0.])
    X,y = np.array(X), np.array(y)
    X,y = shuffle(X, y)
    return X,y

# hot code by depentence with previous amino acids
def protDepHotCode(seq):
    aas = "ACDEFHIGKLMNQPRSTVWYX"
    x = np.zeros(shape=(len(seq), 21))
    for i in range(len(seq)):
        if seq[i] == '#':
                x[i] = np.ones(shape=(21,))*0.05
        else:
            for j in range(21):  
                k,d = -1,0.
                while True:
                    try:
                        k = seq[:i+1].index(aas[j],k+1)
                        d += 2**(k-i)
                    except:
                        break
                x[i,j] = d 
    return x

def protOneHotCode(seq):
    aas = "ACDEFHIGKLMNQPRSTVWYX"
    x = np.zeros(shape=(len(seq), 21))
    for i in range(len(seq)):
        if seq[i] == '#':
                x[i] = np.ones(shape=(21,))/21
        else:
            t = np.zeros((21,))
            t[aas.index(seq[i])]=1
            x[i] = t
    return x
# hot code by migrating model
def protMigratedCode(seq):
    aas = "ACDEFHIGKLMNQPRSTVWYX"
    x = np.zeros(shape=(len(seq), 22))
    v = np.zeros(shape=(22,))
    for i in range(len(seq)):
        if seq[i] == '#':
                x[i] = np.zeros(shape=(22,))
                x[i,21] = 1 
        else:
            j = aas.index(seq[i])
            t = np.zeros(shape=(22,))
            t[j] = 1
            if i == 0:
                x[i] = (t + v)/2
            else:
                x[i] = (x[i-1] + t)/2
            x[i,21] = 0
    return x


# hot code by migrating model
def protMigratedCode2(seq):    
    x = np.zeros(shape=(len(seq), 6))
    v = np.ones(shape=(5,)) * 0.5
    for i in range(len(seq)):
        if seq[i] == '#':
                x[i] = np.zeros(shape=(6,))
                x[i,5] = 1 
        else:
            t = []
            for c in XiaoInfoCode[seq[i]]:
                t.append(eval(c))
            t = np.array(t, dtype=float)
            
            if i == 0:
                x[i,:5] = (v + t)/2
            else:
                x[i,:5] = (x[i-1,:5] + t)/2
            x[i,5] = 0
    return x 
   
def protsFormulateByChaosCode(lsseq):
    X = []
    for seq in lsseq:
        X.append( protMigratedCode(seq))
    return np.array(X)

def protsFormulateByXiaoInfoCode(lsseq):
    X = []
    for seq in lsseq:
        X.append( protMigratedCode2(seq))
    return np.array(X)

def protsFormulateByOneHotCode(lsseq):
    X = []
    for seq in lsseq:
        X.append( protOneHotCode(seq))
    return np.array(X)
#pseqs, psites = readPDNA62()                
#posseqs, negseqs = getTrainingDataset(pseqs, psites, 11)                
#pseqs, psites = readPDNA224()
#posseqs, negseqs = getTrainingDataset(pseqs,psites,11)
"""seq = 'MKLAAHLPH'
seq2 = 'KLAAHLPHQ'
M = protMigratedCode(seq)
M2 = protMigratedCode(seq2)
"""
      

