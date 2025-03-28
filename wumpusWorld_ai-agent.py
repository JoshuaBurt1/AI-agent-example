#To run: solara run a3_jburt4_part_b.py

import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from mesa.datacollection import DataCollector
from mesa.experimental.cell_space import CellAgent, FixedAgent
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def compute_wumpus(model):
    agent_wumpus = int(sum(agent.isWumpus and not agent.dead for agent in model.agents))
    return agent_wumpus
def compute_heroes(model):
    agent_hero = int(sum(not agent.isWumpus and not agent.dead and not agent.isGlitter and not agent.isGold and not agent.isPit and not agent.isBreeze and not agent.isStench for agent in model.agents))                         
    return agent_hero

#The WumpusHero (hero agent) moves according to BGS information
#The hero agent only has room for 1 gold in inventory
class WumpusWorldAgent(CellAgent):
    """An agent with a Wumpus, hero, pits, and gold."""

    def __init__(self, model, cell=None):
        # Pass the parameters to the parent class.
        super().__init__(model)
        self.cell = cell

        # Create the agent's variable and set the initial values.
        self.performanceMeasure = 0

        self.initialPrompt = "Explore the cave. Choices: sense glitter = 4. If you have gold, go to start and choose = 5. Move = 1. Turn = 2 left or 3 right. Avoid breeze and stench unless there is gold. The last thing you write is your choice. "
        self.SENSORstring = ""                                                                                                                  # updated in wumpus_world()
        self.MOVEMENT_REMINDERstring = "If no glitter found yet, choose 1 or 2 and 3 less often. If breeze choose 2 twice in a row, then 1. "   # Initial reponse weight modifier
        self.GLITTER_REMINDERstring = ""                                                                                                        # updated in wumpus_world() on glitter sense 
        self.MEMORYstring = ["You are currently at coordinate (0,0). "]                                                                         # updated in move(), glitter_response(), and return_home()
        self.GOLD_REMINDERstring = ""                                                                                                           # updated in glitter_response() on gold acquisition
        #self.BREEZE_REMINDERstring = ""
        #self.HOME_COORDINATEstring = ""

        self.turn = 3                                                                                                                           # hero faces "up" at start
        self.dead = False        

        self.isWumpus = False
        self.isStench = False
        self.senseStench = False
        self.shots_left = 1

        self.isPit = False
        self.isBreeze = False
        self.senseBreeze = False

        self.isGold = False
        self.isGlitter = False
        self.senseGlitter = False
        self.goldGrabbed = False
        self.inventory = None
        self.goldReleased = False

        self.performanceMeasure = 0

        """
        #This can be used to essentially tell the SLM what to do based on a precise calculation. It is not implemented, but could solve this faster
        #search the area
        self.protocol_map = [0][0]
        #if breeze and no gold
        self.protocol_turn_around = [2,2,1]
        #if you have gold, return home -> convert this to a weight for the prompt to choose a direction
        self.protocol_return_home = [0][0]
        #additional choice weight to go home (0,0)
        self.RETURNGOLDweight = 0
        """

        
    def step(self):
        self.move()                                         # ACTUATORS : LLM SHOULD CONTROL THIS (add a parameter of a parsed self.SENSORstring = "")
        self.place_stench()
        self.wumpus_world()                                 # SENSORS update
        if(not self.isWumpus and not self.dead):            # This is the "NotImplementedHeroAgent"
            #print("BGS LLM string: " + self.SENSORstring)      # SENSORYstring data should be added to MEMORYstring ##### SENSORYstring + MEMORYstring  ########
            self.MEMORYstring[-1] += self.SENSORstring
            self.MEMORYstring[-1] += self.MOVEMENT_REMINDERstring
            self.MEMORYstring[-1] += self.GLITTER_REMINDERstring
            self.MEMORYstring[-1] += self.GOLD_REMINDERstring
            #print("TOTAL MEMORIES: " + self.MEMORYstring)  

    #ACTUATORS
    #this selects to go turn either: left, right, or go forward randomly. Note: an agent does not move forward every turn
    def move(self):
        if(self.isWumpus and not self.dead):
            wumpusMovementChoice = random.randint(1,3) 
            if (wumpusMovementChoice == 1):
                self.turn_left()
            elif (wumpusMovementChoice == 2):
                self.turn_right()
            else:
                self.forward()
        if(not self.isWumpus and not self.dead):
            parsed_response = 0  # the hero must choose where to move
            #print("BGS SENSORY INPUT: " + self.MEMORYstring[-1])  
            self.RECENTMEMORIES = ""
            self.RECENTMEMORIES = self.initialPrompt
            for i in self.MEMORYstring:
                self.RECENTMEMORIES+=i
            if(self.performanceMeasure % 3 == 0):
                self.MEMORYstring.clear()
                self.RECENTMEMORIES = self.initialPrompt + self.SENSORstring + self.MOVEMENT_REMINDERstring + self.GLITTER_REMINDERstring + self.GOLD_REMINDERstring
            print("BGS SENSORY INPUT: " + self.RECENTMEMORIES) 
         
            #print("INITIAL PROMPT:" + self.MEMORYstring[-1])
            #print("TOTAL MEMORIES: ")
            #for i in self.MEMORYstring:
            #    print(i)
            unparsedResponse = self.model.PromptModel(self.initialPrompt, self.RECENTMEMORIES, " Make decisions weighted on risk and reward in a maximum of 20 words, followed by a manditory numeric choice: ")
            print("INTERNAL CONVERSATION: " + unparsedResponse) # starting from the end obtain the first number

            reversed_response = unparsedResponse[::-1]
            for char in reversed_response:
                if char.isdigit():
                    parsed_response = int(char)
                    break
                else:
                    parsed_response = 6
            print("Hero choice: " + str(parsed_response) + "; Performance measure: " + str(self.performanceMeasure) + "\n")

            if (parsed_response == 1):
                self.forward()
                if(not self.isWumpus and not self.dead):
                    self.MEMORYstring.append("You moved forward, currently at " + str(self.pos) + ". ")
                    return self.MEMORYstring
            elif (parsed_response == 2):
                self.turn_right()
                if(not self.isWumpus and not self.dead):
                    self.MEMORYstring.append("You orientated 90 degrees right at " + str(self.pos) + ". ")
                    return self.MEMORYstring
            elif (parsed_response == 3 ):
                self.turn_left()
                if(not self.isWumpus and not self.dead):
                    self.MEMORYstring.append("You orientated 90 degrees left at " + str(self.pos) + ". ")
                    return self.MEMORYstring
            elif (parsed_response == 4 ):
                self.glitter_response()
            elif (parsed_response == 5 ):
                self.return_home()  
            else:
                self.MEMORYstring.append("I didn't move. Maybe I should pick 1, 2, 3, 4 or 5. ")

            #reduce the MEMORYstring if size becomes an issue

    def turn_left(self):
        if not self.dead:
            self.turn+=1
            #if not self.isWumpus:
            #    print(f"Hero {self.model.agents.index(self)} turned left.")
            #else:
            #    print(f"Wumpus {self.model.agents.index(self)} turned left.")  

    def turn_right(self):
        if not self.dead:
            self.turn-=1
            #if not self.isWumpus:
            #    print(f"Hero {self.model.agents.index(self)} turned right.")
            #else:
            #    print(f"Wumpus {self.model.agents.index(self)} turned right.")

    def forward(self):
        if not self.dead:
            x, y = self.pos
            if (self.turn%4==0):
                new_position = (x - 1, y)
                #if not self.isWumpus:
                #    print(f"Hero {self.model.agents.index(self)} went left.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went left.")     
            if (self.turn%4==1):
                new_position = (x, y - 1)
                #if not self.isWumpus:
                #    print(f"Hero {self.model.agents.index(self)} went down.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went down.")
            if (self.turn%4==2):
                new_position = (x + 1, y)
                #if not self.isWumpus:
                #    print(f"Hero {self.model.agents.index(self)} went right.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went right.") 
            if (self.turn%4==3):
                new_position = (x, y + 1)
                #if not self.isWumpus:
                #    print(f"Hero {self.model.agents.index(self)} went up.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went up.")  
            self.model.grid.move_agent(self, new_position)

    def glitter_response(self):
        if(not self.isWumpus and not self.dead):
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            glitter_agent = [agent for agent in cellmates if isinstance(agent, Glitter)]
            gold_agent = [agent for agent in cellmates if isinstance(agent, Gold)]
            #if self.LLM string contains glitter
            if glitter_agent and not self.goldGrabbed:
                self.goldReleased = False
                self.goldGrabbed = True
                self.inventory = glitter_agent
                if glitter_agent:
                    glitter_agent = glitter_agent[0]
                    self.model.grid.remove_agent(glitter_agent)
                self.inventory = gold_agent

                self.MEMORYstring.append("You picked up gold at " + str(self.pos) + ". ")
                self.GOLD_REMINDERstring += "You have gold, go to (0,0) and choose 5. "
                self.GLITTER_REMINDERstring = ""    #Unnecessary
                #self.RETURNGOLDweight += 1
                #print(f"This gold {self.inventory} has been grabbed")
                if gold_agent:
                    gold_agent = gold_agent[0]
                    self.model.grid.remove_agent(gold_agent)
                    #print("Gold has been removed from the grid.")
                return self.GLITTER_REMINDERstring 

            elif glitter_agent and self.goldGrabbed:
                self.MEMORYstring.append("You already have gold. You need to bring it back to position (0,0). You are currently at " +str(self.pos) + ". ")
                #self.RETURNGOLDweight += 1
            elif not glitter_agent and self.goldGrabbed:
                self.MEMORYstring.append("No gold here because you sensed no glitter. You need to bring it back to position (0,0). You are currently at " +str(self.pos) + ". ")
                #self.RETURNGOLDweight += 1
            else:
                self.MEMORYstring.append("You tried to pick up gold but it wasn't in the area at " + str(self.pos) + ". ")

    def return_home(self):
        if(not self.isWumpus and not self.dead and self.pos == (0,0) and self.goldGrabbed):
            self.goldReleased = True
            print(f"Returned to position 0,0 with {self.inventory}")
            self.performanceMeasure += 1000
            gold_agent = Gold(self.model)
            glitter_agent = Glitter(self.model) 
            gold_agent.model.grid.place_agent(gold_agent, (self.pos))
            glitter_agent.model.grid.place_agent(glitter_agent, (self.pos))
            gold_agent.isWumpus = False
            gold_agent.isBreeze = False
            gold_agent.isStench = False
            gold_agent.isPit = False
            gold_agent.dead = False
            gold_agent.shots_left = 0
            glitter_agent.isWumpus = False
            glitter_agent.isBreeze = False
            glitter_agent.isStench = False
            glitter_agent.isPit = False
            glitter_agent.dead = False
            glitter_agent.shots_left = 0
            glitter_agent.isGlitter = True
            glitter_agent.isGold = False
            gold_agent.isGlitter = False
            gold_agent.isGold = True
            self.inventory = None
            self.goldGrabbed = False
            self.MEMORYstring.append("You brought gold back to " + str(self.pos) + ". ")
        elif (not self.isWumpus and not self.dead and not self.pos == (0,0) and self.goldGrabbed):
            self.MEMORYstring.append("You have gold, but are not at the exit. ")
        elif (not self.isWumpus and not self.dead and self.pos == (0,0) and not self.goldGrabbed):
            self.MEMORYstring.append("You do not gold, but are at the exit. ")
        else:
            self.MEMORYstring.append("You are at " + str(self.pos) + ", but have no gold. ")

    

    def place_stench(self):
        if(self.isWumpus):
            stench_cells = self.model.grid.get_neighborhood(
                self.pos,
                moore=False,  # Only the 4 orthogonal neighbors
                include_center=False
            )        
            for cell in stench_cells:
                stench_agent = Stench(self.model)
                self.model.grid.place_agent(stench_agent, cell)
                stench_agent.isStench = True
                stench_agent.isWumpus = False
                stench_agent.isPit = False
                stench_agent.dead = False
                stench_agent.shots_left = 0
                stench_agent.isGold = False
                stench_agent.isGlitter = False
                stench_agent.isBreeze = False

    def wumpus_world(self):
        self.SENSORstring = ""
        others = self.model.grid.get_cell_list_contents([self.pos])
        for other in others:              
            if(self.isWumpus and not self.dead and not other.isWumpus and not other.dead and not (other.isGlitter or other.isGold or other.isBreeze or other.isStench or other.isPit)):             
                other.performanceMeasure -= 1000
                other.dead = True
                print(f"Hero {self.model.agents.index(other)} is dead from Wumpus.")   
                return
                                        
            if(not self.isWumpus and not self.dead and other.isWumpus and self.shots_left > 0): 
                self.shots_left -= 1  
                self.performanceMeasure -=10
                print(f"Wumpus {self.model.agents.index(other)} dead. Hero {self.model.agents.index(self)} shots left: {self.shots_left}")          
                other.dead = True   
                other.isWumpus = False
                other.shots_left = 0
                return
            #run out of arrows
            if(not self.isWumpus and not self.dead and other.isWumpus and self.shots_left == 0): 
                print(f"Arrow gone. Hero {self.model.agents.index(self)} has {self.shots_left} arrows remaining.") 
                return 
            #pit
            if(not self.isWumpus and not self.dead and other.isPit):
                print(f"Hero {self.model.agents.index(self)} found pit {self.model.agents.index(other)}")  
                self.performanceMeasure -= 1000
                self.dead = True
                if(self.dead):
                    print(f"Hero {self.model.agents.index(self)} is dead from falling into pit {self.model.agents.index(other)}")  
                return
            
            if(not self.isWumpus and not self.dead and other.isStench):
                #print("Stench true")
                self.senseStench = True
            if(not self.isWumpus and not self.dead and other.isBreeze):
                #print("Breeze true")
                self.senseBreeze = True
            if(not self.isWumpus and not self.dead and other.isGlitter):
                #print("Glitter true")
                self.senseGlitter = True
            if(not self.isWumpus and not self.dead and not other.isStench):
                #print("Stench false")
                self.senseStench = False
            if(not self.isWumpus and not self.dead and not other.isBreeze):
                #print("Breeze false")
                self.senseBreeze = False
            if(not self.isWumpus and not self.dead and not other.isGlitter):
                self.senseGlitter = False
                #print("Glitter false")
            
            if (self.senseStench):
                self.SENSORstring += "You sensed a stench at " +str(self.pos) + ". "
            elif (self.senseBreeze):
                self.SENSORstring += "You sensed a breeze at " +str(self.pos) + ". "
            elif (self.senseGlitter):
                self.SENSORstring += "You sensed glitter at " +str(self.pos) + ". "
                self.GLITTER_REMINDERstring += "You sensed glitter at " +str(self.pos) + ". Choose 4. "
                if(not self.inventory == None):
                     self.GLITTER_REMINDERstring = ""
                self.MOVEMENT_REMINDERstring = ""    #Unnecessary

                
        if(not self.isWumpus and not self.dead):
            #print("LLM string: " + self.SENSORstring)
            self.performanceMeasure -=1
            return self.SENSORstring
        return 
            

class WumpusWorldModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, heroNumber=1, wumpusNumber=1, goldNumber=2, pitNumber=1, width=4, height=4): # CHANGED : percentWumpus=10
        super().__init__()
        self.hero_number = heroNumber
        self.wumpus_number = wumpusNumber
        self.gold_number = goldNumber
        self.pit_number = pitNumber
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Heroes": compute_heroes, "Wumpus": compute_wumpus},agent_reporters={"Wealth": "wealth"}
        )
        # Create agents 
        for i in range(self.wumpus_number):
            wumpus_agent = WumpusWorldAgent(self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            cellmates = self.grid.get_cell_list_contents([(x, y)])
            while (len(cellmates) >0 or (x, y) == (0,0)):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                cellmates = self.grid.get_cell_list_contents([(x, y)]) 
            self.grid.place_agent(wumpus_agent, (x, y))
            wumpus_agent.isWumpus = True
            wumpus_agent.dead = False
            wumpus_agent.shots_left = 0
            wumpus_agent.isGold = False
            wumpus_agent.isStench = False
            wumpus_agent.isPit = False
            wumpus_agent.isBreeze = False
            wumpus_agent.isGlitter = False
            stench_cells = self.grid.get_neighborhood(
                wumpus_agent.pos,
                moore=False,
                include_center=False
            )        
            for cell in stench_cells:
                stench_agent = Stench(self)
                self.grid.place_agent(stench_agent, cell)
                stench_agent.isStench = True
                stench_agent.isWumpus = False
                stench_agent.isPit = False
                stench_agent.dead = False
                stench_agent.shots_left = 0
                stench_agent.isGold = False
                stench_agent.isGlitter = False
                stench_agent.isBreeze = False

        for i in range(self.hero_number):
            hero_agent = WumpusWorldAgent(self)
            hero_agent.isWumpus = False
            hero_agent.dead = False
            hero_agent.shots_left = 1
            hero_agent.isGold = False
            hero_agent.isPit = False
            hero_agent.isGlitter = False
            hero_agent.isBreeze = False
            hero_agent.isStench = False
            self.grid.place_agent(hero_agent, (0, 0))

        for i in range(self.gold_number): 
            gold_agent = Gold(self)
            glitter_agent = Glitter(self) 
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            cellmates = self.grid.get_cell_list_contents([(x, y)])
            while (len(cellmates) >0):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                cellmates = self.grid.get_cell_list_contents([(x, y)]) 
            self.grid.place_agent(gold_agent, (x, y))
            self.grid.place_agent(glitter_agent, (x, y))
            gold_agent.isWumpus = False
            gold_agent.isBreeze = False
            gold_agent.isStench = False
            gold_agent.isPit = False
            gold_agent.dead = False
            gold_agent.shots_left = 0
            glitter_agent.isWumpus = False
            glitter_agent.isBreeze = False
            glitter_agent.isStench = False
            glitter_agent.isPit = False
            glitter_agent.dead = False
            glitter_agent.shots_left = 0
            glitter_agent.isGlitter = True
            glitter_agent.isGold = False
            gold_agent.isGlitter = False
            gold_agent.isGold = True

        for i in range(self.pit_number):
            pit_agent = Pit(self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            cellmates = self.grid.get_cell_list_contents([(x, y)])
            while (len(cellmates) >0):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                cellmates = self.grid.get_cell_list_contents([(x, y)]) 
            self.grid.place_agent(pit_agent, (x, y))
            pit_agent.isWumpus = False
            pit_agent.dead = False
            pit_agent.shots_left = 0
            pit_agent.isGold = False
            pit_agent.isStench = False
            pit_agent.isGlitter = False
            pit_agent.isBreeze = False
            pit_agent.isPit = True
            breeze_cells = self.grid.get_neighborhood(
                    pit_agent.pos,
                    moore=False,
                    include_center=False
                )
            for j in breeze_cells:
                breeze_agent = Breeze(self)
                self.grid.place_agent(breeze_agent, j)
                breeze_agent.isWumpus = False
                breeze_agent.isPit = False
                breeze_agent.dead = False
                breeze_agent.shots_left = 0
                breeze_agent.isGold = False
                breeze_agent.isStench = False
                breeze_agent.isGlitter = False
                breeze_agent.isBreeze = True

        self.turn_into_wumpus() # CHANGED
        self.running = True
        #self.datacollector.collect(self)

        ################ LLM CODE ################
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_properties(i))

        torch.random.manual_seed(0)

        model_path = "microsoft/Phi-4-mini-instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0", # "cpu" or "auto" or "cuda:0" for cuda device 0, 1, 2, 3 etc. if you have multiple GPUs
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        #gotta have a tokenizer for each model otherwise the token mappings won't match
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def PromptModel(self, context, memorystream, prompt):
        
        #lower temperature generally more predictable results, you can experiment with this
        generation_args = {
            "max_new_tokens": 64,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": True,
        }

        llmprompt = prompt
        
        messages = [
            {"role": "system", "content": context},
            {"role": "system", "content": memorystream},
            {"role": "user", "content": llmprompt},
        ]

        #time1 = int(round(time.time() * 1000))

        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']

        #time2 = int(round(time.time() * 1000))
        #print("Generation time: " + str(time2 - time1))
        #self.datacollector.collect(self)
    ####################################################

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
    
    def turn_into_wumpus(self):     
        num_wumpus = int(self.wumpus_number)    
        print(f"Total wumpus agents: {num_wumpus}")    
        for i in range(num_wumpus):                    
            self.agents[i].isWumpus = True              
        print(f"Total wumpusHero agents: {self.hero_number}")   

        
model_params = {
    "wumpusNumber": {
        "type": "SliderInt",
        "value": 1,   # matched to initial settings, this can change on user RESET
        "label": "Wumpus #:",
        "min": 1,
        "max": 100,
        "step": 1,
    },
    #UPGRADE 2: % Wumpus slider: this changes the percentage of agents as wumpus on the grid after sliding the bar and pressing reset.
    "heroNumber": {
        "type": "SliderInt", 
        "value": 1,    # matched to initial settings, this can change on user RESET
        "label": "Hero #:", 
       "min": 1,    
        "max": 100,  
        "step": 1,  
    },
    "goldNumber": {
        "type": "SliderInt", 
        "value": 2,    # matched to initial settings, this can change on user RESET
        "label": "Gold #:", 
       "min": 1,    
        "max": 100,  
        "step": 1,  
    },
    "pitNumber": {
        "type": "SliderInt", 
        "value": 1,    # matched to initial settings, this can change on user RESET
        "label": "Pit #:", 
       "min": 1,    
        "max": 100,  
        "step": 1,  
    },
    "width": {
        "type": "SliderInt",
        "value": 4,    # matched to initial settings, this can change on user RESET
        "label": "Width:",
        "min": 4,
        "max": 40,
        "step": 4,
    },
    "height": {
        "type": "SliderInt",
        "value": 4,    # matched to initial settings, this can change on user RESET
        "label": "Height:",
        "min": 4,
        "max": 40,
        "step": 4,
    },
}

class Gold(FixedAgent):
    """A static gold bar that can be picked up by the hero."""
    def __init__(self, model):
        """Create a new gold bar.
        Args:
            model: Model instance
        """
        super().__init__(model)

class Glitter(FixedAgent):
    """A visual associated with gold."""
    def __init__(self, model):
        """Create a new gold bar.
        Args:
            model: Model instance
        """
        super().__init__(model)

class Pit(FixedAgent):
    """A pit that ends the heros journey."""
    def __init__(self, model):
        """Create a new pit.
        Args:
            model: Model instance
        """
        super().__init__(model)

class Breeze(FixedAgent):
    """A breeze that is always in a pit's neighborhood."""
    def __init__(self, model):
        """Create a new breeze.
        Args:
            model: Model instance
        """
        super().__init__(model)

class Stench(FixedAgent):
    """A stench that is always in a wumpus's neighborhood."""
    def __init__(self, model):
        """Create a new stench.
        Args:
            model: Model instance
        """
        super().__init__(model)
        self.turns_remaining = 1
    def step(self):
        self.turns_remaining -= 1
        if self.turns_remaining <= 0:
            if self.pos is not None:
                self.model.grid.remove_agent(self)

#modify this function to change output on grid
def agent_portrayal(agent):
    size = 20
    color = "tab:red" 
    if isinstance(agent, Glitter):  # Gold representation
        size = 60
        color = "gold"
    
    if isinstance(agent, Pit):  # Pit representation
        size = 60
        color = "black"
    
    if isinstance(agent, Breeze):  # Breeze representation
        size = 5
        color = "cyan"
        
    if agent.isStench:  # Breeze representation
        size = 5
        color = "pink"  

    if agent.isWumpus:  # Wumpus representation
        size = 20
        color = "tab:green" 
 
    if agent.dead:  # Dead agent representation
        size = 20
        color = "tab:orange"
    
    return {"size": size, "color": color}

ww_model = WumpusWorldModel(1, 1, 2, 1, 4, 4) # v. When you create your agents, set 10% of them to be Wumpus

def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

SpaceGraph = make_space_component(agent_portrayal)

def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))

WumpusHeroPlot = make_plot_component(
    {"Wumpus": "tab:green", "Heroes": "tab:red"},
    post_process=post_process_lines
)

page = SolaraViz(
    ww_model,
    components=[SpaceGraph, WumpusHeroPlot],
    model_params=model_params,
    name="Wumpus World Model"
)

# This is required to render the visualization in the Jupyter notebook
page