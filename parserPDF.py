# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 08:55:34 2019

@author: Administrator
"""

from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

def parsePdf2Txt(pdfFile, txtFile):
    fp = open(pdfFile,'rb')
    parser = PDFParser(fp)
    document = PDFDocument()
    parser.set_document(document)
    document.set_parser(parser)
    document.initialize()
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr,device)
    for page in document.get_pages():
        interpreter.process_page(page)
        layout = device.get_result()
        for x in layout:
            if (isinstance(x, LTTextBoxHorizontal)):
                with open(txtFile, 'a', encoding='utf-8' ) as f:
                    result = x.get_text()
                    f.write(result + '\n')
                
if __name__ == "__main__":
    pdfFile = r'PDNA_Data\Supp1-Pho_S.pdf'
    txtFile = r'PDNA_Data\Supp1-Pho_s.txt'
    parsePdf2Txt(pdfFile, txtFile)