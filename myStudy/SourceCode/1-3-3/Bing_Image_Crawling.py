# From Microsoft Bing or Baidu, Crawling Images 
# py -m pip install icrawler

from icrawler.builtin import BingImageCrawler
# crawler = BingImageCrawler(storage={"root_dir": "dogs"})
crawler = BingImageCrawler(storage={"root_dir": "cats"})
crawler.crawl(keyword="cat", max_num=10)