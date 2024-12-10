import requests
import json, csv
from datetime import datetime
import time, calendar
from pytz import timezone
import ConfigParser, os, sys
import logging

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger( __name__ )

logger = logging.getLogger()
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

config = ConfigParser.ConfigParser()
config.read('trello.ini')

key = config.get('TRELLO', 'key')
token = config.get('TRELLO', 'token')
thelist = config.get('TRELLO', 'list')
startdatefield = config.get('CUSTOM', 'startdate')
durationfield = config.get('CUSTOM', 'duration')
tempoidfield = config.get('CUSTOM', 'tempoid')
daysfield = config.get('CUSTOM', 'period')

def getCardsbyMember(member):
    url = "https://api.trello.com/1/members/%s" % (member)
    querystring = {"fields":"username,fullName",
    "cards":"open","card_fields":"name,idList",
    "key":key,
    "token":token}
    headers = {
    'cache-control': "no-cache",
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = json.loads(response.text)
    return data

def getTempoData(cardid):
    url = "https://api.trello.com/1/cards/%s/pluginData" % (cardid)
    querystring = {"key":key,
    "token":token}
    headers = {
    'cache-control': "no-cache",
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.text
    j_data = data[1:len(data)-1]
    if len(j_data) > 0:
        return json.loads(j_data)
    else:
        return "NULL"

def writeToCSV(tempoid,startdate,duration,description):
    data = [tempoid, startdate, duration, description]
    f = open(str(filedate)+'.csv','a')
    w = csv.writer(f)
    w.writerow(data)
    f.close()

# http://stackoverflow.com/questions/466345/converting-string-into-datetime
def getWorklogDate(period,date):
    # remove the timezone from the string which come from trello
    date=date[0:len(date)-6]
    struct_time = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    #struct_time = datetime.strptime("2017-02-10T12:00:00", "%Y-%m-%dT%H:%M:%S")
    struct_time.strftime('%w')
    datetime_obj_hk = timezone('Asia/Hong_Kong').localize(struct_time)
    n = int(period)
    date_arr = []
    while n >= 1:
        date_arr.append(datetime_obj_hk)
        weekofday=int(datetime_obj_hk.strftime('%w'))+1
        # unix time 86400 per day
        unix_datetime = calendar.timegm(datetime_obj_hk.utctimetuple())
        # 0 = sunday
        if weekofday == 7:
            unix_datetime = unix_datetime + 172800
        # 6 = Sat
        if weekofday == 6:
            unix_datetime = unix_datetime + 259200
        datetime_obj_hk = timezone('Asia/Hong_Kong').localize(datetime.fromtimestamp(unix_datetime))
        n=n-1;
    return date_arr

def getCompleteCard(cards):
    completedcard = []
    for card in cards:
        # find the card if it save in ##Completed
        if card["idList"] == thelist:
            cardname = card["name"]
            # get the custom field
            tempo_d = getTempoData(card["id"])
            # if there are TEMPO data
            if tempo_d != "NULL":
                try:
                    d = json.loads(tempo_d["value"])
                    startdate = d["fields"][startdatefield]
                    duration = int(d["fields"][durationfield])*60*60
                    tempoid = d["fields"][tempoidfield]
                    period = d["fields"][daysfield]
                    date_arr = getWorklogDate(period,startdate)
                    for d in date_arr:
                        myDict = {"tempoid": tempoid, "startdate": d, "duration": duration, "cardname":cardname}
                        completedcard.append(myDict)
                    logger.info("added %s into csv" % (cardname))
                except KeyError:
                    logger.error("Skip %s because error" % (cardname))
                    logger.debug(sys.exc_info())
                    pass
                except:
                    print "Unexpected error:", sys.exc_info()[0]
            else:
                logger.warning("No TEMPO data in %s" % (cardname))
    return completedcard

filedate = datetime.today().strftime("%Y-%m-%d")
member = sys.argv[1] if len(sys.argv) > 1 else "539010c187655c291d15ef13"
d=getCardsbyMember(member)
cards=getCompleteCard(cards=d["cards"])
for card in cards:
    writeToCSV(card["tempoid"],card["startdate"],card["duration"],card["cardname"])
