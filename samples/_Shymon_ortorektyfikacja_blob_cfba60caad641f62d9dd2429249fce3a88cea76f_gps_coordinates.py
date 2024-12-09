# TODO: Wzorowane na https://stackoverflow.com/questions/10852955/python-batch-convert-gps-positions-to-lat-lon-decimals
import utm

def convert_wgs(wgs_string):
  """ Converts wgs_string to decimal value of latitude or longitude """
  direction = {'N':1, 'S':-1, 'E': 1, 'W':-1}
  new = wgs_string.strip().replace(u'deg',' ').replace('\'',' ').replace('"',' ')
  new = new.split()
  new_dir = new.pop()
  new.extend([0,0,0])
  return (float(new[0])+float(new[1])/60.0+float(new[2])/3600.0) * direction[new_dir]

def to_utm(latitude, longitude):
  # Returns (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
  return utm.from_latlon(latitude, longitude)

# Przykładowe użycie:
# lat, lon = u''' 50 deg 25' 16.72" N, 18 deg 45' 2.83" E '''.split(', ')
# print(conversion(lat), conversion(lon))