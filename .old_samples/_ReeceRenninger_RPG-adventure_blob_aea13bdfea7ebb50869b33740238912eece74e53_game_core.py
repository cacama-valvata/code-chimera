#! /usr/bin/python3
# Base file for the game contains the main game code
# Ctrl + f5 to run the game
# import time module to delay text, this is needed to manipulate the speed of the text
import time
import sys
import random
import json
from character import Character, Items, Weapon, Barbarian, Paladin, Ranger, choose_character_class


#** create sample character
#step One: create character and items
player = Character("Starter Character")


#** delay print function
def print_delay(text,delay):
    print(text, end='', flush=True)
    time.sleep(delay)
    print()

delay_time = 0.5

#** Load scenarios json
def load_scenarios(file_path="scenarios.json"):
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    return scenarios

#** single character delay print, https://stackoverflow.com/questions/9246076/how-to-print-one-character-at-a-time-on-one-line
def slow_print(string):
    for char in string: # for each character in the string
        sys.stdout.write(char) # writes the character to the console using write method of sys.stdout
        sys.stdout.flush() # after each character is written, flush the output buffer to immediately display the character rather than waiting for the buffer to fill
        time.sleep(0.05) # this is the delay time between each character, change this to change the speed of the text (.05 is fast)

#** validator function for user input
def valid_user_input(valid_choices):
    while True:
        choice = input().lower() # case sensitive change to prevent errors
        if choice in valid_choices:
            return choice
        else:
            print_delay("Please enter a valid choice from your options given!", delay_time)

def game_intro():
    intro_text = "Welcome to the game!"
    print_delay(intro_text, 0.5)
    print_delay("This is a text based adventure game where you will be given a series of choices to make.", delay_time)
    print_delay("These choices will determine the outcome of the game.", delay_time)
    print_delay("Good luck!", delay_time)

# calls the scenarios function with choices saved under scenarios.json, trying to extract 
# information from main game file so its not so bloated
def main_game_logic(player):
    scenarios = load_scenarios() #load scenarios from json
    play_scenario("start", player, scenarios)

def play_scenario(scenario_key, play, scenarios):
    scenario = scenarios[scenario_key] #how we grab scenarios based off player key input passed
    print_delay(scenario["text"], delay_time)

    print_delay(f"Welcome, {player.name}!", 1)
    print(f"You've chosen the {type(player).__name__} class. You will start with {player.health} health.")
    if player.weapon:
        print(f"The {type(player).__name__} class will have access to {player.weapon.name} it has a damage range of {player.weapon.damage_range}.")
       
    else:
        print("You have no weapon equipped.")
    print_delay(f"Your adventure is about to begin {player.name}, prepare yourself", 1)
    slow_print(f"You awaken in a dark cave, you have no memory of how you got here. As you stand up, you realize there is a small light coming from a distant corner of the cave. As you begin to move toward the light, 
               you see your {player.weapon.name} leaning against the side of the cave near the light. You pick up {player.weapon.name} and continue into the light.  As you emerge from the cave, you see a path leading into a forest. You begin to walk down the path...")

    # This is the first choice the player will make
    #!! continue the introduction to be more in depth and figure out if I can delay the text to be more like reading an actual story
    print_delay("You are walking down a path and you come to a fork in the road.", delay_time)
    print_delay("Do you go left or right?", delay_time)
    print_delay("Type 'left' or 'right' and press enter to choose.", delay_time)

    #! CURRENTLY ONLY BUILDING THE LEFT 
    valid_path_choices = ["left", "right"]
    pathChoice = valid_user_input(valid_path_choices)

    # This is the first outcome of the game
    if pathChoice == "left":
        print_delay("You chose to go left, you see the path continue deeper into forest.", delay_time)
        print_delay("You continue down the path and you come to a clearing.", delay_time)
        print_delay("You see a small house in the clearing.", delay_time)
        print_delay("Do you go inside the house or continue down the path?", delay_time)
        print_delay("Type 'house' or 'path' and press enter to choose.", delay_time)
        houseChoice = valid_user_input(["house", "path"])

    if houseChoice == "house":
        print_delay("You head towards the door of the house and check if it is unlocked.", delay_time)
        print_delay("The door is unlocked and you enter the house.", delay_time)
        goblin_health = 5
        print_delay("You are immediately attacked by a goblin!", delay_time)
        print_delay(f"You pull out your {player.weapon.name} and strike the goblin!", delay_time)
        print_delay(f"You deal {player_damage} damage to the goblin!", delay_time)
        if player_damage >= goblin_health:
            print_delay("You have defeated the goblin!", delay_time)
        elif player_damage < goblin_health:
            print_delay(f"The goblin has {goblin_health} health remaining!", delay_time)
            def goblin_dmg():
                goblin_damage = random.randint(1, 4)
                return goblin_damage
            print_delay(f"The goblin strikes back at you! He strikes you for {goblin_dmg()} damage!")
            print_delay(f"Your total health is now {player.health - goblin_dmg()}")
            print_delay(f"You strike back at the goblin! Dealing {player_damage} damage!")
            if goblin_health == 0:
                print_delay("You have defeated the goblin!")
    
    elif houseChoice == "path":
        print_delay("You decide to avoid the house in the clearing and continue down the path.", delay_time)
        print_delay("You see the path begins to narrow and enter into the forest.", delay_time)
        print_delay("As you enter the forest you hear a rustling in the bushes.", delay_time)
        print_delay("Do you investigate the bushes or continue deeper into the forest?", delay_time)
        print_delay("Type 'investigate' or 'forest' and press enter to choose.", delay_time)
        forestChoice = input()

    elif pathChoice == "right":
        print_delay("As you continue down the path you see a clearing off in the distance.", delay_time)
        print_delay("As your vision turns back to the path, you see a cart flipped over.", delay_time)
        print_delay("Do you investigate the cart or continue on the path?.", delay_time)
        print_delay("Type 'investigate' or 'continue' and press enter to choose.", delay_time)
        cartChoice = valid_user_input(["investigate", "continue"])

    if cartChoice == "investigate":
        print_delay("You chose to investigate the cart.", delay_time)
        print_delay("You walk up to the cart and see a dead body laying next to it.", delay_time)
        print_delay("You search the body and find a small pouch of gold.", delay_time)
        print_delay("You turn back and start continue heading down the path.", delay_time)
        print_delay("You come to a clearing and see a small house in the clearing.", delay_time)
        print_delay("Do you go inside the house or continue down the path?", delay_time)
        print_delay("Type 'house' or 'path' and press enter to choose.", delay_time)
        secondHouseChoice = valid_user_input(["path", "house"]) #!! can I recycle the house choice from above or do I need to create a new one?

    elif cartChoice == "continue":
        print_delay("You decide to avoid the cart and continue walking.", delay_time)
        print_delay("You come to a clearing and see a small house in the clearing.", delay_time)
        print_delay("")


    
    #if forestChoice == "investigate":

    #elif forestChoice == "forest":
        




# indentation of this is very important or it will all break
if __name__ == "__main__":
    game_intro()

    #!! global player variable
    player = choose_character_class()

    #!! global dmg roll for player
    player_damage = player.weapon.roll_damage()

    main_game_logic(player)