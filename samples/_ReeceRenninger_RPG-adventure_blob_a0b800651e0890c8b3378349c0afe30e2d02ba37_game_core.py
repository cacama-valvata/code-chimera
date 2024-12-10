# Base file for the game contains the main game code

# import time module to delay text, this is needed to manipulate the speed of the text
import time
import sys
from character import Character, Items, Weapon, Barbarian, Paladin, Ranger, choose_character_class

# create sample character

#step One: create character and items
player = Character("Starter Character")


# delay print function
def print_delay(text,delay):
    print(text, end='', flush=True)
    time.sleep(delay)
    print()

delay_time = 0.5

# single character delay print, https://stackoverflow.com/questions/9246076/how-to-print-one-character-at-a-time-on-one-line
def delay_print(string):
    for char in string:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05) # this is the delay time between each character, change this to change the speed of the text (.05 is fast)


def game_intro():
    intro_text = "Welcome to the game!"
    print_delay(intro_text, 0.5)
    print_delay("This is a text based adventure game where you will be given a series of choices to make.", delay_time)
    print_delay("These choices will determine the outcome of the game.", delay_time)
    print_delay("Good luck!", delay_time)

def main_game_logic(player):
    ##!! trying to figure out to get player dmg rolls

    print_delay(f"Welcome, {player.name}!", 1)
    print(f"You've chosen the {type(player).__name__} class. You will start with {player.health} health.")
    # if player.weapon:
    #     print(f"You have a {player.weapon.name} equipped.")
       
    # else:
    #     print("You have no weapon equipped.")
    print_delay(f"Your adventure is about to begin {player.name}, prepare yourself", 1)
    delay_print(f"You awaken in a dark cave, you have no memory of how you got here. As you stand up, you realize there is a small light coming from a distant corner of the cave. As you begin to move toward the light, you see a {player.weapon.name} leaning against the side of the cave near the light. You pick up {player.weapon.name} and continue into the light.  As you emerge from the cave, you see a path leading into a forest. You begin to walk down the path...")
    # This is the first choice the player will make
    #!! continue the introduction to be more in depth and figure out if I can delay the text to be more like reading an actual story
    print("You are walking down a path and you come to a fork in the road.")
    print("Do you go left or right?")
    print("Type 'left' or 'right' and press enter to choose.")
    pathChoice = input()

    # if pathChoice != "left" or pathChoice != "right":
    #     print("You must choose left or right.")
    #     print("Type 'left' or 'right' and press enter to choose.")
    #     pathChoice = input()

    # This is the first outcome of the game
    if pathChoice == "left":
        print("You chose to go left, you see the path continue deeper into forest.")
        print("You continue down the path and you come to a clearing.")
        print("You see a small house in the clearing.")
        print("Do you go inside the house or continue down the path?")
        print("Type 'house' or 'path' and press enter to choose.")
        houseChoice = input()

    elif pathChoice == "right":
        print("You chose to go right, you see the path continue toward a clearing.")
        print("You see a cart flipped on its side on the path.")
        print("Do you investigate the cart or continue on the path?.")
        print("Type 'investigate' or 'path' and press enter to choose.")
        cartChoice = input()

    if houseChoice == "house":
        print("You head towards the door of the house and check if it is unlocked.")
        print("The door is unlocked and you enter the house.")
        goblin_health = 5
        print("You are immediately attacked by a goblin!")
        print(f"You pull out your {player.weapon.name} and strike the goblin!")
        print(f"You deal {player_damage} damage to the goblin!")
        if player_damage >= goblin_health:
            print("You have defeated the goblin!")
        else: 
            print(f"The goblin has {goblin_health} health remaining!")

# indentation of this is very important or it will all break
if __name__ == "__main__":
    game_intro()

    #!! global player variable
    player = choose_character_class()

    #!! global dmg roll for player
    player_damage = player.weapon.roll_damage()

    main_game_logic(player)