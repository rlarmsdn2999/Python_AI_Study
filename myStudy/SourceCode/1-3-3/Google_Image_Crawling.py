# Set google Chrome Browser. Check the file of 'chromedriver.exe' is in the same folder. 
# Searching and collecting the images with a keyword from Google Chrom browser
# Collecting the all images with scrolling from the browser

from selenium import webdriver              # py -m pip install selenium
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

driver = "chromedriver.exe" # webdriver.Chrome()
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ogbl")  # Searching Google Images 
elem = driver.find_element_by_name("q")                         # search window
####################### Set the keyword for searching ############################
elem.send_keys("xylobot")                             ### Input a keyword for searching
##################################################################################
elem.send_keys(Keys.RETURN)                           ### Enter

SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height: # if scrolled to the bottom, 
        try:                      # click the button of 'more images'
            driver.find_element_by_css_selector(".mye4qd").click()
        except:                   # if no more image, search will be finished
            break
    last_height = new_height

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count = 1               # set the name of image file to 1,2,3...
for image in images:
    try:
        image.click()   # select the 1st image
        time.sleep(3)   # sleep 3 seconds for searching of site
        imgURL = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")
        ################## Set the path for saving the images to your PC ####################
        # Change the below folder path to the path for your own PC 
        outpath = "C:/HyACT/Python Project/Image Datasets/Google Crawling/" # Your PC's folder
        ##################################################################################### 
        outfile = str(count) + ".jpg"
        urllib.request.urlretrieve(imgURL, outpath+outfile)
        count = count + 1
    except:
        pass

driver.close()    # close driver