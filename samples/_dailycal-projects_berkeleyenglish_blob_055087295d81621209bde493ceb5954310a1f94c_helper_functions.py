from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import string

def google_search(search_term, **kwargs):
    """ Searches google for given argument 
    Not sure what is happening? Me neither.
    Source here: 
    https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search
    """
    service = build("customsearch", "v1", developerKey=my_api_key)
    res = service.cse().list(q=search_term, cx=my_cse_id, **kwargs).execute()
    return res['items']

def find_gender(url):
    """ Returns string "male", "female" or "other" for given wikipedia page.
    Gender determined on the number of "he/him/his", "she/her/hers", or
    "they/their/them" on the first introductory paragraphs of the page.
    """
    soup = open_url(url)
    p_tag = soup.find('p')

    number_male, number_female, number_other = 0, 0, 0
    while p_tag.name == 'p':
        content = p_tag.get_text().lower()
        content.maketrans('', '', string.punctuation)
        words = content.split()
        number_male += words.count("he") + words.count("him") + words.count("his")
        number_female += words.count("she") + words.count("her") + words.count("hers")
        number_other += words.count("they") + words.count("theirs") + words.count("them")
        p_tag = p_tag.next_sibling.next_sibling # wikipedia puts \n as the next sibling,
        # so find the one after that.

    maximum = max(number_male, number_female, number_other)
    if maximum == number_male and maximum != number_female: 
        return 'male'
    elif maximum == number_female and maximum != number_male:
        return 'female'
    else:
        return 'other'

def open_url(url):
    response = requests.get(url)
    html = response.content

    soup = BeautifulSoup(html, 'html.parser')
    return soup

# CONSTANTS
my_api_key = "AIzaSyAeg1Eq9YPMrlktNkdwIzkOm7lFj1XPilk"
my_cse_id = "013182980831658096104:b_duphwes6c"

