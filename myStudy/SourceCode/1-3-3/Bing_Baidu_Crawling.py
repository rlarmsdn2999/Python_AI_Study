# From Microsoft Bing or Baidu or Google, Crawling Images with Keywords
# py -m pip install icrawler

from icrawler.builtin import BaiduImageCrawler 
from icrawler.builtin import BingImageCrawler 
from icrawler.builtin import GoogleImageCrawler 
# Input the key-word for Crawling
list_word = ['抽烟 行人','玩手机 行人']

for word in list_word:
    ########### Bing Crawling ############
    # Path to save the crawling images
    bing_storage = {'root_dir': 'bing\\'+word}

    bing_crawler = BingImageCrawler(parser_threads=2,
                                    downloader_threads=4,
                                    storage=bing_storage)
    # starting crawling with keyword and maximum number of crawling 
    bing_crawler.crawl(keyword=word,
                       max_num=5)

    ############ Baidu Crawling ###########
    # baidu_storage = {'root_dir': 'baidu\\' + word}
    # baidu_crawler = BaiduImageCrawler(parser_threads=2,
    #                                   downloader_threads=4,
    #                                   storage=baidu_storage)
    # baidu_crawler.crawl(keyword=word,
    #                     max_num=2000)


    ############ Google Crawling ###########
    # google_storage = {'root_dir': '‘google\\' + word}
    # google_crawler = GoogleImageCrawler(parser_threads=4,
    #                                    downloader_threads=4,
    #                                    storage=google_storage)
    # google_crawler.crawl(keyword=word,
    #                      max_num=2000)
