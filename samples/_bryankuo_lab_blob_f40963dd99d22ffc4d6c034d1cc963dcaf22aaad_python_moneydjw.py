#!/usr/bin/python3

# python3 moneydjw.py
# exercise selenium tips on exception, element picking
# \param in
# \param out
# return 0

import sys, requests, time, os, numpy, random, csv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import ElementNotSelectableException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import ElementNotInteractableException

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from timeit import default_timer as timer
from datetime import timedelta,datetime
from pprint import pprint

# url = "https://www.moneydj.com/warrant/xdjhtm/default.xdjhtm"

# portal
# url = "https://www.moneydj.com/warrant/xdjhtm/default.xdjhtm"

# rank
url = "https://www.moneydj.com/warrant/xdjhtm/Rank.xdjhtm?a=01"
# url = "https://www.moneydj.com/warrant/xdjhtm/Rank.xdjhtm?a=02"
# url = "https://www.moneydj.com/warrant/xdjhtm/Rank.xdjhtm?a=03"
# url = "https://www.moneydj.com/warrant/xdjhtm/Rank.xdjhtm?a=04"

# url = "https://www.moneydj.com/warrant/xdjhtm/Quote.xdjhtm"

print("fetch {}".format(url))

try:
    browser = webdriver.Safari(executable_path = '/usr/bin/safaridriver')
    # @see https://stackoverflow.com/a/26567563
    browser.implicitly_wait(20) # when .get()
    browser.maximize_window()
    browser.switch_to.window(browser.current_window_handle)
    browser.get(url)

    page1 = browser.page_source
    soup = BeautifulSoup(page1, 'html.parser')

    options = soup.find_all("select", {"class": "pageselect"})[0] \
        .find_all("option")
    print("# opt (pages) {}".format(len(options)))

    delay = 2
    # WebDriverWait(browser, delay) \
    #    .until(EC.presence_of_element_located(By.Class, "clearBoth datalist"))

    # WebDriverWait(browser, delay) \
    #    .until(EC.visibility_of_element_located(By.Class, "clearBoth datalist"))

    time.sleep(delay)

    tables = soup.find_all("table", {"class": "clearBoth datalist"})
    print("# tables {}".format(len(tables)))

    rows = tables[0].find_all("tr")
    print("# rows {}".format(len(rows)))

    # lis = soup.find_all("div", {"id": "rkTab"})[0] \
    #     .find_all("ul")[0] \
    #    .find_all("li")
    # print("# lis {}".format(len(lis))) # works

    # element = browser.find_element(By.LINK_TEXT, '成交金額排行') # works
    # element =  lis[1].find_all("a")[0] # // FIXME: not triggering
    # element = browser \
    #    .find_element(By.LINK_TEXT, lis[2].find_all("a")[0].text)
    # print("click {}".format(lis[2].find_all("a")[0].text)) # works
    # action = webdriver.ActionChains(browser)
    # action.click(element).perform() # works

    print("+click {}".format(options[1].text)) # works
    action = webdriver.ActionChains(browser)
    # action.click(options[2]).perform() # // FIXME

    # Select(WebDriverWait(browser, 3).until(                 \  // FIXME
    #    EC.element_to_be_clickable((                        \
    #        By.XPATH,"//select[@class='pageselect']"))))    \
    #            .select_by_value(options[2].text)
    #            # .select_by_value('2')
    #            # .select_by_value(options[2].value)

    # pageselect = Select(WebDriverWait(browser, 3)                     \
    #    .until(EC.element_to_be_clickable(                          \
    #        (By.XPATH, \
    #    "//*[@id=\"RankDiv\"]/div/div[1]/div[1]/table/tbody/tr/td[3]/select"))))

    # pageselect = browser.findElement(
    #    By.xpath("//*[@id=\"RankDiv\"]/div/div[1]/div[1]/table/tbody/tr/td[3]/select"))
    # pageselect.selectByValue("2")

    # // TODO:
    pageselect = \
    Select(WebDriverWait(browser, 10) \
        .until(EC.element_to_be_clickable( \
            (By.XPATH, \
    "//*[@id=\"RankDiv\"]/div/div[1]/div[1]/table/tbody/tr/td[3]/select"))))
    print("{}".format(pageselect))
    print([o.text for o in pageselect.options])
    pageselect.select_by_visible_text(pageselect.options[1].text)

    # @see https://stackoverflow.com/a/29059348
    # pageselect.select_by_value(options[1].text)
    #action.move_to_element(pageselect)
    #pageselect.select_by_value('2')
    print("-click {}".format(options[1].text))

    # reload soup before write to file?
    page1 = browser.page_source
    soup = BeautifulSoup(page1, 'html.parser')

    html_path = "moneydjw.html"
    outfile2 = open(html_path, "w")

    outfile2.write(soup.prettify())
    outfile2.close()
    print("write to {}".format(html_path))

# So your Automation Script may be facing any of these exceptions:
# @see https://stackoverflow.com/a/53589205
except NoSuchElementException:
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    pass
    # raise

except TimeoutException:
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    pass

except ElementNotVisibleException:
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    pass

except ElementNotSelectableException:
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    pass

except ElementClickInterceptedException:
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    pass

except ElementNotInteractableException:
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    pass

except SessionNotCreatedException:
    print('turn on safari remote option.')

except:
    # traceback.format_exception(*sys.exc_info())
    e = sys.exc_info()[0]
    print("Unexpected error:", sys.exc_info()[0])
    raise

finally:
    browser.minimize_window() # OK
    browser.quit()

sys.exit(0)
