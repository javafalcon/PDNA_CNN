# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:57:47 2019

@author: Administrator
"""
pdna_seqs_62 = {}
pdna_sites_62 = {}
with open(r'H:\Mywork\btt029_Supplementary_Data\Supplementary_data_S3_010913.doc','r',encoding='GB18030') as fr:
    n = 0
    lseq = []
    key = ""
    lsite = []
    start = False
    for line in fr:
        line = line.strip("\n")
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
            k = label.index(".") 
            key = label[:k]
        elif start:
            ls = line.split(" ")
            if ls[0] == '-':
                lsite.append( ls[1])
            lseq.append(ls[2])
    
    pdna_seqs_62[key] = "".join(lseq)
    pdna_sites_62[key] = lsite        
            