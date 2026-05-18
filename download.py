#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Download pdf list according to a .txt file (each line is a pdf url)

import os, sys
file_list = sys.argv[1] # the file name of the txt file
with open(file_list, 'r', encoding='utf-8') as f:
    pdf_urls = f.readlines()
for pdf_url in pdf_urls:
    # Allowing annotate for saved pdf name
    # example:
    # http://www.moe.gov.cn/srcsite/A26/s8001/202204/W020220420582343475848.pdf #test.pdf
    # then the saved pdf name will be test.pdf
    if '#' in pdf_url:
        pdf_url, pdf_name = pdf_url.split('#')
    else:
        pdf_name = pdf_url.split('/')[-1]
    pdf_name = pdf_name.strip()
    if pdf_name:
        os.system(f"wget {pdf_url} -O {pdf_name}")
    
    