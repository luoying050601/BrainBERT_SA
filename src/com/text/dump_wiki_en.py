# import requests
# # Library for parsing HTML
# from bs4 import BeautifulSoup
base_url = 'https://dumps.wikimedia.org/enwiki/'
# index = requests.get(base_url).text
# soup_index = BeautifulSoup(index, 'html.parser')
# # Find the links on the page
# dumps = [a['href'] for a in soup_index.find_all('a') if
#          a.has_attr('href')]
# print(dumps)
# dump_url = base_url + '20201201/'
# # Retrieve the html
# dump_html = requests.get(dump_url).text
# # Convert to a soup
# soup_dump = BeautifulSoup(dump_html, 'html.parser')
# # Find list elements with the class file
# var = soup_dump.find_all('li', {'class': 'file'})[:3]
# print(var)
# [<li class="file">
# <a href="/enwiki/20201201/enwiki-20201201-pages-articles-multistream.xml.bz2">
# enwiki-20201201-pages-articles-multistream.xml.bz2</a> 17.7 GB</li>,
# <li class="file">
# <a href="/enwiki/20201201/enwiki-20201201-pages-articles-multistream-index.txt.bz2">
# enwiki-20201201-pages-articles-multistream-index.txt.bz2</a> 217.3 MB</li>,
# <li class="file"><a href="/enwiki/20201201/enwiki-20201201-pages-articles-multistream1.xml-p1p41242.bz2">enwiki-20201201-pages-articles-multistream1.xml-p1p41242.bz2</a> 233.1 MB</li>]
# ['../', '20201201/', '20201220/', '20210101/', '20210120/', '20210201/', '20210220/', '20210301/', 'latest/']
#

from keras.utils import get_file
import subprocess
saved_file_path = get_file('enwiki-20201201-pages-articles-multistream.xml.bz2',
                           base_url+'/20201201/enwiki-20201201-pages-articles-multistream.xml.bz2')
data_path = '/home/ying/.keras/datasets/enwiki-20201201-pages-articles-multistream.xml.bz2'
# Iterate through compressed file one line at a time
for line in subprocess.Popen(['bzcat'],
                              stdin = open(data_path),
                              stdout = subprocess.PIPE).stdout:
    print(line)
    # process line
