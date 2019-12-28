# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:58:57 2019

@author: falcon1
"""
from  AminoAcidCode import *
import numpy as np

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

"""
def aaseq2naseq(seq):
    seq = seq.upper()
    seq = re.sub('[ZUB]','',seq)
    seq = seq.strip()
    naseq = ""
    for c in seq:
        naseq += aa2code(c) 
    return naseq
"""
"""
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
"""

# 每个氨基酸编码为5位0、1序列。返回len(seq)-by-5的二维数组
def protXiaoInfoCode(seq):
    x = np.zeros(shape=(len(seq), 6))
    for i in range(len(seq)):
        if seq[i] == '#':
            x[i] = np.zeros(shape=(6,))
            x[i,5] = 1
        else:
            c = code['XiaoInfoCode'][seq[i]]
            for k in range(5):
                x[i,k] = int(c[k])
            x[i,5] = 0
        
    return x

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

# One-Hot编码.前21列是21个氨基酸（含X）出现则在对应的列置为1.最后1列是标识位
# 如果是填充的列则置为1，否则为0
# 返回len(seq)-by-22的二维数组
def protOneHotCode(seq):
    aas = "ACDEFHIGKLMNQPRSTVWYX"
    x = np.zeros(shape=(len(seq), 22))
    for i in range(len(seq)):
        if seq[i] == '#':
                x[i,:] = np.zeros(shape=(22,))
                x[i,21] = 1
        else:
            t = np.zeros((22,))
            t[aas.index(seq[i])]=1
            t[21] = 0
            x[i] = t
    return x

""" 
 One Hot编码的游走形式编码。
 对蛋白质序列seq中的每一个氨基酸残基首先进行One-Hot编码, seq[i]对应到一个21维向量x[i](i=1,2,...)，
 然后：x[i] = x[i] + x[i-1]
 置X[0] = [0,0,0,...,0]
 返回序列的二维数组表示。len(seq)-by-22的二维数组
"""
def protMigratedOneHotCode(seq):
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

"""
 XiaoInfoCode的游走形式编码。
 与protMigrateOneHotCode相同，只是置x[0] = [0.5,0.5,...,0.5]
 返回len(seq)-by-6的二维数组，最后一列是标志是否为填充位的标识
"""
def protMigratedXiaoInfoCode(seq):    
    x = np.zeros(shape=(len(seq), 6))
    v = np.ones(shape=(5,)) * 0.5
    for i in range(len(seq)):
        if seq[i] == '#':
                x[i] = np.zeros(shape=(6,))
                x[i,5] = 1 
        else:
            t = []
            for c in code['XiaoInfoCode'][seq[i]]:
                t.append(eval(c))
            t = np.array(t, dtype=float)
            
            if i == 0:
                x[i,:5] = (v + t)/2
            else:
                x[i,:5] = (x[i-1,:5] + t)/2
            x[i,5] = 0
    return x 
 
"""
 蛋白质序列的理化属性编码
"""    
def protPhychemCode(seq):
    x = np.zeros(shape=(len(seq), 8))
    for i in range(len(seq)):
        if seq[i] == '#':
            x[i] = np.zeros(shape=(8,))
            x[i,7] = 1
        else:
            x[i,:7] = code['phychemCodeLog'][seq[i]]
            x[i,7] = 0
        
    return x


def protsFormulateByChaosCode(lsseq:list):
    X = []
    for seq in lsseq:
        X.append( protMigratedCode(seq))
    return np.array(X)

def protsFormulateByXiaoInfoCode(lsseq:list):
    X = []
    for seq in lsseq:
        X.append( protXiaoInfoCode(seq))
    return np.array(X)

def protsFormulateByOneHotCode(lsseq:list):
    X = []
    for seq in lsseq:
        X.append( protOneHotCode(seq))
    return np.array(X)

def protsFormulateByPhychemCode(lsseq:list):
    X = []
    for seq in lsseq:
        X.append( protPhychemCode(seq))
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
      

