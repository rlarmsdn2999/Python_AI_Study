######## Recommended Crawling Python Program ##########
# From Microsoft Bing or Baidu or Google, Crawling Images with Keywords
# py -m pip install icrawler

from icrawler.builtin import BingImageCrawler
# from icrawler.builtin import BaiduImageCrawler, GoogleImageCrawler

############## Keyword and max_num Setting ##############
key_word = input('Keyword: ')       # keyword for searching, ex) dog, cat, horse....
crawl_num = int(input('Max_num of Crawling: '))  # Maxim number for crawling  ex) 1000

############## Crawling from Bing #############
bing_crawler = BingImageCrawler(
    feeder_threads = 1,
    parser_threads = 1,
    downloader_threads = 4, 
    storage = {'root_dir':'iCrawler\\'+key_word} ) # current directory/iCrawler/key_word

bing_crawler.crawl(keyword=key_word, filters=None, offset=0, max_num=crawl_num)

############## Crawling from Baidu #############
# baidu_crawler = BaiduImageCrawler(
#     feeder_threads = 1,
#     parser_threads = 1,
#     downloader_threads = 4, 
#     storage = {'root_dir':'Baidu\\'+key_word} )

# baidu_crawler.crawl(keyword=key_word, offset=0, max_num=crawl_num, 
#                         min_size=(200, 200), max_size=None)

############## Crawling from Google #############
# google_crawler = GoogleImageCrawler(
#     feeder_threads = 1,
#     parser_threads = 1,
#     downloader_threads = 4, 
#     storage = {'root_dir':'GoogleCrawling\\'+key_word} 
#     )
# # filters = dict(
# #     size = 'large',
# #     color = 'orange',
# #     license = 'commercial, modify', 
# #     # date = ((2020.1.1), (2021.8.30))
# #     )

# google_crawler.crawl(keyword=key_word, filters=None, offset=0, max_num=crawl_num, 
#                         min_size=(200, 200), max_size=None, file_idx_offset=0)
print('Image Crawling is done.')