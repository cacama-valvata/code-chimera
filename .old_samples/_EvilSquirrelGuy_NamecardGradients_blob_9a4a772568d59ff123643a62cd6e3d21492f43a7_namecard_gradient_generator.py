import PIL.Image
import numpy as np
import bs4
import cv2
import pickle
import win32api, win32con

from colorama import Fore, Back, Style
from urllib.request import Request, urlopen

NAMECARD_URL = "https://antifandom.com/genshin-impact/wiki/Namecard"
# after h2 span#Navigation


def get_url_contents(url: str) -> bs4.BeautifulSoup:
  """
  Gets the contents of the url and parses it into a BeautifulSoup element.
  """

  req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
  doc = urlopen(req).read()
  page = bs4.BeautifulSoup(doc, "html.parser")

  return page


def get_navigation_table(doc: bs4.BeautifulSoup) -> bs4.Tag:
  """
  Gets the navigation table from the page.
  """
  # get the table after a h2 with ID Navigation with the navbox-border class
  nav_table = doc.select("h2:has(span#Navigation) ~ table.navbox-border")

  table = [el for el in nav_table][0]

  return table


def get_images_from_element(element: bs4.Tag) -> dict[str, str]:
  """
  Gets all the images in the element as a dictionary of alt: src
  """

  images = {
    img.get("alt"): img.get("src").split("/scale-to-width-down")[0]
      for img in element.find_all("img")
    }

  return images


def load_images_to_dict(image_dict: dict[str, str]) -> dict[str, cv2.typing.MatLike]:
  """
  Gets the images from the provided urls and turns them into objects.
  """

  try:
    print(Style.DIM + f"  Attempting to read image data from cached data..." + Style.RESET_ALL)
    with open(".namecards_bin.pkl", "rb") as pkl:
      return pickle.load(pkl)
  except:
    print(Style.RESET_ALL + Fore.RED + "  Failed to read data from cache, falling back to parsing...")

  result = dict()

  for name in image_dict:
    print(Style.DIM + f"  Loading image for {name}..." + Style.RESET_ALL + " "*20, end="\r")
    req = urlopen(image_dict[name])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    result[name] = img

  with open(".namecards_bin.pkl", "wb") as pkl:
    win32api.SetFileAttributes(".namecards_bin.pkl", win32con.FILE_ATTRIBUTE_HIDDEN)
    print(Style.RESET_ALL + Fore.YELLOW + f"  Caching binary image data..." + " "*20)
    pickle.dump(result, pkl)

  return result


def get_dominant_colours(img: cv2.typing.MatLike, count: int = 1) -> tuple | list:
  """
  Gets the `count` dominant colours in a cv2 image.
  """
  # reshape the rgb image into an array
  #print(img[0])

  # from: https://stackoverflow.com/questions/3241929/how-to-find-the-dominant-most-common-color-in-an-image

  pixels = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
  im_pil = PIL.Image.fromarray(pixels)

  paletted = im_pil.convert("P", palette=PIL.Image.ADAPTIVE, colors=count)

  palette = paletted.getpalette()
  colour_counts = sorted(paletted.getcolors(), reverse=True)
  palette_index = colour_counts[0][1]
  
  dominant_colours = []

  for i in range(count):
    palette_index = colour_counts[i][1]
    dominant_colours.append(palette[palette_index*3:palette_index*3+3])

  # count is the number of clusters

  #bestLabels = np.zeros((pixels.shape[0], 1), dtype=np.int32)

  # perform k-means clustering
  # kmeans = cv2.kmeans(
  #   pixels.astype(np.float32), count,
  #   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0),
  #   flags=cv2.KMEANS_RANDOM_CENTERS,
  #   bestLabels=bestLabels,
  #   attempts=3
  # )

  # dominant_colours = kmeans[2].astype(np.uint32)

  return dominant_colours


def rgb_tuple_to_hex(rgb: tuple) -> str:
  """
  Converts a tuple of (r, g, b) into a hex code string
  """
  return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb_tuple(hex_code: str) -> tuple:
  """
  Converts a hex code string to an rgb tuple.
  """
  # strip any starting #
  if hex_code.startswith("#"):
    hex_code = hex_code[1:]

  r = int(hex_code[:2], base=16)
  g = int(hex_code[2:4], base=16)
  b = int(hex_code[4:], base=16)

  return r, g, b


def formatted_colour(hex_code: str = None, rgb_tuple: tuple = None, text: str = None):
  """
  Prints a colour in the given rgb
  """
  if hex_code is not None and rgb_tuple is None:
    red, green, blue = hex_to_rgb_tuple(hex_code)
  elif rgb_tuple is not None:
    red, green, blue = rgb_tuple
  else:
    raise ValueError("Function requires at least 1 valid colour.")

  if not text:
    if hex_code is not None:
      if not hex_code.startswith("#"):
        text = f"#{hex_code}".lower()
      else:
        text = hex_code.lower()
    else:
      text = rgb_tuple_to_hex(rgb_tuple)

  return "\x1B[38;2;{};{};{}m{}\x1B[0m".format(red, green, blue, text)


def colour_distance(start: tuple, end: tuple) -> float | int:
  """
  Gets the distance between 2 colours.
  """
  return (start[0] - end[0])**2 + (start[1] - end[1])**2 + (start[2] - end[2])**2


def generate_gradient_from_colours(
    start: tuple,
    end: tuple,
    colours: list,
    steps: int = 16
) -> tuple:
  """
  Generates a gradient from start to end using only the predetermined colours
  defined in the colours list, with the specified amount of steps.
  """

  gradients = []

  steps -= 1  # correct the steps value to get the total number of colours we need

  # find the closest start and end colours
  closest_start = min(colours, key=lambda colour: colour_distance(start, colour))
  closest_end = min(colours, key=lambda colour: colour_distance(end, colour))

  # get the step size
  step_size_r = (closest_end[0] - closest_start[0]) / steps
  step_size_g = (closest_end[1] - closest_start[1]) / steps
  step_size_b = (closest_end[2] - closest_start[2]) / steps

  # generate the gradient
  gradients.append(closest_start)

  for i in range(1, steps):
    r = int(closest_start[0] + step_size_r * i)
    g = int(closest_start[1] + step_size_g * i)
    b = int(closest_start[2] + step_size_b * i)

    # find the closest colour
    closest_colour = min(
      colours,
      key = lambda colour: colour_distance((r, g, b), colour)
    )

    if closest_colour not in gradients and closest_colour not in [closest_start, closest_end]:
      gradients.append(closest_colour)

  gradients.append(closest_end)

  return gradients


def generate_similar_colours(theme_colour: tuple, colours: list, top: int = 16):
  """
  Generates a list of colours that are the closest to the provided theme colour.
  """

  colours.sort(key=lambda colour: colour_distance(theme_colour, colour))

  # all the elements in the array
  if len(colours) < top:
    top = len(colours)

  return sorted(colours[:top], key=lambda colour: colour_distance((0,0,0), colour))



def main_gradient(colours: list):
  """
  The coroutine for generating a gradient from 1 colour to another.
  """

  start_colour = input(
    Fore.GREEN + Style.BRIGHT +
    "Enter gradient start colour: #" +
    Style.NORMAL
  )
  
  end_colour = input(
    Fore.RED + Style.BRIGHT +
    "Enter gradient end colour: #" +
    Style.NORMAL
  )

  steps = int(input(
    Fore.YELLOW + Style.BRIGHT +
    "Enter required gradient steps: " +
    Style.NORMAL
  ))

  print(Style.BRIGHT + Fore.BLUE + "Your Choices:")

  print(
    Style.NORMAL + Fore.BLACK + Back.WHITE + "Gradient Start Colour: " + 
    formatted_colour(hex_code=start_colour)
  )
  
  print(
  Style.NORMAL + Fore.BLACK + Back.WHITE + "Gradient End Colour: " + 
  formatted_colour(hex_code=end_colour)
  )
  
  start = hex_to_rgb_tuple(start_colour)
  end = hex_to_rgb_tuple(end_colour)

  gradient = generate_gradient_from_colours(
    start, end, colours, steps
  )

  print()
  
  print(Back.CYAN + Style.BRIGHT + Fore.BLUE + "Resultant Gradient:" + Style.RESET_ALL)

  return gradient


def main_similar(colours: list):
  """
  The coroutine for generating a list of namecards closest to a colour.
  """

  theme_colour = input(
    Style.RESET_ALL + Fore.YELLOW + Style.BRIGHT +
    "Enter theme colour: #" +
    Style.NORMAL
  )

  top = int(input(
    Fore.CYAN + Style.BRIGHT +
    "Enter how many namecards you want: " +
    Style.NORMAL
  ))

  print(Style.BRIGHT + Fore.BLUE + "Your Choices:")

  print(
    Style.NORMAL + Fore.BLACK + Back.WHITE + "Gradient Start Colour: " + 
    formatted_colour(hex_code=theme_colour)
  )

  theme = hex_to_rgb_tuple(theme_colour)

  cards = generate_similar_colours(theme, colours, top)

  print()

  print(Back.CYAN + Style.BRIGHT + Fore.BLUE + "Resultant Namecards:" + Style.RESET_ALL)

  return cards




def main(*args) -> None:
  """
  Main program function.
  """
  print(Back.BLUE + Fore.WHITE + "Welcome to the Genshin Namecard Gradient Generator" + Style.RESET_ALL)

  print("Reading Webpage...")
  namecard_page = get_url_contents(NAMECARD_URL)
  print("Extracting element...")
  nav_table = get_navigation_table(namecard_page)
  print("Extracting images from element...")
  namecard_urls = get_images_from_element(nav_table)
  print("Loading images into memory...")
  namecard_bin = load_images_to_dict(namecard_urls)
  print()

  print("Extracting dominant colours from images...")
  # get the dominant colours from the image
  dominant_colours_by_img = {
    name: get_dominant_colours(namecard_bin[name], 1)[0]
      for name in namecard_bin
  }
  
  # print(dominant_colours_by_img)

  colours = list(dominant_colours_by_img.values())

  print(f"""{Style.BRIGHT + Fore.GREEN}Available modes:
{Style.NORMAL+Fore.CYAN}  1. Similar Namecards
{Style.NORMAL+Fore.BLUE}  2. Namecard Gradient
{Style.NORMAL+Fore.RED}  q. Exit{Style.RESET_ALL}""")

  mode = input(f"{Style.BRIGHT+Fore.BLUE}Select a mode: {Style.DIM}")
  
  if mode == "1":
    result = main_similar(colours)
  elif mode == "2":
    result = main_gradient(colours)  
  elif mode == "q":
    exit(0)
  else:
    raise ValueError("Invalid mode!")

  for colour in result:
    # print each colour in the gradient formatted
    print(
      "- " + formatted_colour(rgb_tuple=colour,
        text=list(dominant_colours_by_img.keys())[
        list(dominant_colours_by_img.values()).index(colour)
      ]) + " - " + rgb_tuple_to_hex(colour)
    )




if __name__ == "__main__":
  main()

