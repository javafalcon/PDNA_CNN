# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:57:47 2019

@author: Administrator
"""
import numpy as np
from tools import aa2code
import re
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
# 读入PDNA-62数据集.返回蛋白质序列及其dna结合位点在序列中的序号（从1开始计数）
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

# 读入PDNA-224数据。返回蛋白质序列及其dna结合位点在序列中的序号（从1开始计数）
# 返回两个字典。
# pdna_seqs_224  序列字典, key：蛋白质id  value: 蛋白质序列
# pdna_sites_224 结合位点字典, key：蛋白质id  value:位点序号组成的列表
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

# 对蛋白质序列进行滑窗，生成正样本和负样本。滑窗尺寸为ws，一个氨基酸左右各取ws个氨基酸，
# 构成一个长度为2*ws+1的肽链，如果中间的氨基酸是与DNA结合的位点，则该序列为正样本
# 否则为负样本。如果氨基酸前后不足ws个残基，则补‘#’       
# 保存滑窗结果到posseqs和negseqs两个列表对象，并保存到npz格式文件中：
# posseqs: list对象，每个元素是长度为2*ws+1的氨基酸序列
# negseqs: list对象，每个元素是长度为2*ws+1的氨基酸序列       
def buildBenchmarkDataset(pseqs:dict, psites:dict, windown_wise:int, npzfile):
    keys = pseqs.keys()
    posseqs, negseqs = [], []
    for key in keys:
        seq = pseqs[key]
        n = len(seq)
        #给序列左右各填充windown_wise个'X'
        seq = list(seq)
        for k in range(windown_wise):
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
    #return posseqs, negseqs
    np.savez(npzfile, pos=posseqs, neg=negseqs)

# 构建用于交叉验证的训练集和测试集序列样本
def generateKFBenchmarkDataset(posseqs:list, negseqs:list, npzfile, kf=5):
    X_train_pos_ls, X_test_pos_ls = [], []
    kf = KFold(n_splits=kf)
    for train_index, test_index in kf.split(posseqs):
        X_train_pos, X_test_pos = [], []
        for k in train_index:
            X_train_pos.append(posseqs[k])
        for j in test_index:
            X_test_pos.append(posseqs[j])
        
        X_train_pos_ls.append(X_train_pos)
        X_test_pos_ls.append(X_test_pos)

    X_train_neg_ls, X_test_neg_ls = [], []
    for train_index, test_index in kf.split(negseqs):
        X_train_neg, X_test_neg = [], []
        for k in train_index:
            X_train_neg.append(negseqs[k])
        for j in test_index:
            X_test_neg.append(negseqs[j])
            
        X_train_neg_ls.append(X_train_neg)
        X_test_neg_ls.append(X_test_neg)
    
    #return (X_train_pos_ls, X_test_pos_ls), (X_train_neg_ls, X_test_neg_ls)
    np.savez(npzfile, trainPos=X_train_pos_ls, trainNeg=X_train_neg_ls, 
             testPos=X_test_pos_ls, testNeg=X_test_neg_ls)

if __name__ == "__main__":
    #benchData = np.load('PDNA_224_11.npz')
    #generateKFBenchmarkDataset(benchData['pos'], benchData['neg'], 'KfBenchmarkDataset.npz') 
    pseqs,psites = readPDNA224()
    buildBenchmarkDataset(pseqs, psites, 20, 'PDNA_224_20.npz')
    benchData = np.load('PDNA_224_20.npz')
    generateKFBenchmarkDataset(benchData['pos'], benchData['neg'], 'KfBenchmarkDataset_20.npz')
    
    
    
    
    