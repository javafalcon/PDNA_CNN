# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:57:47 2019

@author: Administrator
"""
# 处理PDNA-62数据集
def readPDNA62():
    pdna_seqs_62 = {}
    pdna_sites_62 = {}
    with open(r'PDNA_Data\Supplementary_data_S3_010913.doc','r',encoding='ISO-8859-1') as fr:
        n = 0
        lseq = []
        key = ""
        lsite = []
        start = False
        for line in fr:
            line = line.strip("\n")
            if line.startswith(r'.end'):
                break
            if line.startswith(r'.\lab'):
                start = True
                n += 1
                if n > 1:
                    pdna_seqs_62[key] = "".join(lseq)
                    pdna_sites_62[key] = lsite
                    lseq = []
                    lsite = []
                ls = line.split(" ")
                label = ls[0].split("\\")
                label = label[2]
                k = label.rindex(".",0,-1) 
                key = label[:k]
            elif start:
                ls = line.split(" ")
                ind = 1
                
                for s in ls[1:]:
                    if s:
                        if ind == 1:
                            nsite = int(s)
                        elif ind == 2:
                            aa = s
                        ind += 1
                if ls[0] == '-':
                    lsite.append( nsite)
                lseq.append(aa)
        pdna_seqs_62[key] = "".join(lseq)
        pdna_sites_62[key] = lsite
    return pdna_seqs_62, pdna_sites_62

def readPDNA224():
    pdna_seqs_224 = {}
    pdna_sites_224 = {}
    with open(r'PDNA_Data\Supplementary_data_S1_010913.doc','r',encoding='ISO-8859-1') as fr:
        sequencesFlag, sitesFlag = False, False
        for line in fr:
            line = line.replace("\n","")
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
        site = psites[key]
        n = len(seq)
        for i in range(n):
            start = i - windown_wise
            end = i + windown_wise
            if start < 0:
                start = 0
            if end > n-1:
                end = n
            seqseg = seq[start:end]
            if i in site:
                posseqs.append(seqseg)
            else:
                negseqs.append(seqseg)
    return posseqs, negseqs

#pseqs, psites = readPDNA62()                
#posseqs, negseqs = getTrainingDataset(pseqs, psites, 11)                
pseqs, psites = readPDNA224()
posseqs, negseqs = getTrainingDataset(pseqs,psites,11)