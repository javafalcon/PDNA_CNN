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
        site = psites[key]
        n = len(seq)
        for i in range(n):
            start = i- windown_wise
            end = i + windown_wise+1
            if start < 0:
                start = 0
            if end > n-1:
                end = n
            seqseg = seq[start:end]
            if i+1 in site:
                posseqs.append(seqseg)
            else:
                negseqs.append(seqseg)
    return posseqs, negseqs

pseqs, psites = readPDNA62()                
#posseqs, negseqs = getTrainingDataset(pseqs, psites, 11)                
#pseqs, psites = readPDNA224()
#posseqs, negseqs = getTrainingDataset(pseqs,psites,11)