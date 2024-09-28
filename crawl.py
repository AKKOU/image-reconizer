from icrawler.builtin import GoogleImageCrawler
import os


while True:
    # path = input("folder:")
    # os.makedirs(path)   
    google = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': "face1"}
    )
    keyword = input("keyword:")
    google.crawl(keyword=keyword, file_idx_offset=200, max_num=1000, min_size=(200,200), max_size=None)