import os
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from datetime import datetime, date, timedelta
import locale
import pymorphy3
import xlsxwriter
from urllib.parse import urljoin
import re
import math
from alive_progress import alive_bar
import time
import plotext as plt


## Setting locale for later datetime configuration
locale.setlocale(locale.LC_ALL, 'ru_RU')

start_time = datetime.now()
today = date.today()
yesterday = today - timedelta(days = 1)

os.system('cls' if os.name == 'nt' else 'clear')
print('Enter a word to be used')
target_word = input()

os.system('cls' if os.name == 'nt' else 'clear')

## Confirmation
print("Is '", target_word, "' correct? Y/N (Default: Y)" )
confirmation = input()
if confirmation == 'Y':
    url = f'https://tengrinews.kz/search/?text={target_word}'
elif confirmation == 'N':
    print('Enter a word to be used')
    target_word = input()
else:
    url = f'https://tengrinews.kz/search/?text={target_word}'

os.system('cls' if os.name == 'nt' else 'clear')

article_names = []
dates = []
links = []



## First load to get the page number
page = requests.Session()
retries = Retry(total=7, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
page.mount('http://', HTTPAdapter(max_retries=retries))
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

### Base data
farticle_names = []
fdates = []
flinks = []
dates = []
posted_today = []
    
## Page counter/selector
result_number_string = soup.find('p', string=re.compile(r'\d'))
result_number = str(result_number_string).split()[2]
print('Found ', result_number, ' results...')
page_max_num = math.ceil(int(result_number)/20)
print('That will be ', page_max_num, ' page_s')
while True:
    print("Enter the number of pages to work with or use 'A' for all.")
    page_num = input()
    if page_num.isdecimal() != True and page_num != 'A':
            print('There was an error with the input, please try again')
            continue
    else: 
        if page_num == 'A':
            page_num = page_max_num
            break
        elif page_num.isdecimal():
            if int(page_num) <= page_max_num:
                break
            else:
                print('There was an error with the input, please try again')
                continue
            
            

page_num = int(page_num)
os.system('cls' if os.name == 'nt' else 'clear')

## Conjugation and regrouping
m = pymorphy3.MorphAnalyzer()
page_active_num = 1
with alive_bar(page_num) as bar:
    while page_active_num <= page_num: 
        url = f"https://tengrinews.kz/search/page/{page_active_num}/?field=all&text={target_word}&sort=date"
        
        ## Resoup with different page
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        
        ## Raw data scrapping
        content = soup.find('div', class_='content_main')
        titles = content.find_all(class_='content_main_item_title')
        meta = content.find_all('div', class_='content_main_item_meta')
        strdates_old = [strdate.find('span').text.strip() for strdate in meta]
        days, months_old, years = ([] for i in range(3))
        
        ## Splitting the date strings
        for word in strdates_old:
            if page_active_num == 1 and word == 'Сегодня':
                posted_today.append(today)
            elif page_active_num == 1 and word == 'Вчера':
                posted_today.append(yesterday)
            else:     
                day, month, year = word.split(' ')
                days.append(day)
                months_old.append(month)
                years.append(year)

        ## Proper declension
        ## https://ru.stackoverflow.com/questions/767102/strptime-%D0%B4%D0%BB%D1%8F-%D0%BC%D0%B5%D1%81%D1%8F%D1%86%D0%B0-%D0%B2-%D1%80%D0%BE%D0%B4%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%BE%D0%BC-%D0%BF%D0%B0%D0%B4%D0%B5%D0%B6%D0%B5
        months_new = [m.parse(months_old[i])[0].inflect({'nomn'}).word.title() for i in range(len(months_old))]
        strdates_new = [' '.join([days[i], months_new[i], years[i]]) for i in range(len(months_old))]
        article_names = [name.get_text(strip=True) for name in titles]
        dates = [datetime.strptime(strdates_new[date1], '%d %B %Y').date() for date1 in range(len(strdates_new))]
        base_url = 'https://tengrinews.kz/'
        links = [urljoin(base_url, (link.a.get('href'))) for link in titles]
        
        ## Result tracking
        farticle_names = farticle_names + article_names
        if page_active_num == 1:
            fdates = fdates + posted_today
        fdates = fdates + dates
        flinks = flinks + links
        page_active_num = page_active_num + 1  
        
        bar()
    
print(len(farticle_names), 'articles successfully parsed')

data = pd.DataFrame(
    {
        "Article": (farticle_names),

        "Date": (fdates),

        "Link": (flinks),
        
    }
)
print('Dataframe for parsed articles successfully built')

## Word counter

counter_years = [int(i.strftime('%Y')) for i in fdates]

counter_targets = []
counter_results = []

for i in counter_years:
    if i not in counter_targets:
        counter_targets.append(i)

for i in counter_targets:
    counter_results.append(counter_years.count(i))


counter_data = pd.DataFrame(
    {
        "Year": (counter_targets),

        "Frequency": (counter_results),

    }
)
print('Dataframe for counted data successfully built')

## xlxs export
cwd = os.getcwd()
excel_output_data = cwd + fr'\\output\{target_word}_output_data.xlsx'
data.to_excel(excel_output_data)

excel_output_counter_results = cwd + fr'\\output\{target_word}_output_counter_results.xlsx'
counter_data.to_excel(excel_output_counter_results)

## time calculation shi
end_time = datetime.now()

time_taken = end_time - start_time

## no fucking idea 😭
def time_separator(delta):
     days, seconds = delta.days, delta.seconds
     thours = days * 24 + seconds // 3600
     tminutes = (seconds % 3600) // 60
     tseconds = (seconds % 60)
     return thours, tminutes, tseconds
time_separated = (time_separator(time_taken))
## All 3 measurements are not 00
if time_separated[0] != 00 and time_separated[1] != 00 and time_separated[2] != 00:
    time_taken = f"{time_separated[0]}h {time_separated[1]}m {time_separated[2]}s"
## No hour difference
elif time_separated[0] == 00 and time_separated[1] != 00 and time_separated[2] != 00:
    time_taken = f"{time_separated[1]}m {time_separated[2]}s"
## No hour and minute difference
elif time_separated[0] == 00 and time_separated[1] == 00 and time_separated[2] != 00:
    time_taken = f"{time_separated[2]}s"
## No minute difference
elif time_separated[0] != 00 and time_separated[1] == 00 and time_separated[2] != 00:
    time_taken = f"{time_separated[0]}h {time_separated[2]}s"
       
os.system('cls' if os.name == 'nt' else 'clear')
plt.bar(counter_targets, counter_results, color=(135, 95, 255))
plt.title(f"Number of tengrinews.kz articles that include the word {target_word}. {time_taken}")
plt.show()

print('joe biden')