#To run: solara run a3_jburt4_part_a.py

import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from mesa.datacollection import DataCollector
from mesa.experimental.cell_space import CellAgent, FixedAgent
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

def compute_wumpus(model):
    agent_wumpus = int(sum(agent.isWumpus and not agent.dead for agent in model.agents))
    return agent_wumpus
def compute_heroes(model):
    agent_hero = int(sum(not agent.isWumpus and not agent.dead and not agent.isGlitter and not agent.isGold and not agent.isPit and not agent.isBreeze and not agent.isStench for agent in model.agents))                         
    return agent_hero

#The WumpusHero moves randomly.
#The hero agent only has room for 1 gold in inventory
class WumpusWorldAgent(CellAgent):
    """An agent with a Wumpus, hero, pits, and gold."""

    def __init__(self, model, cell=None):
        # Pass the parameters to the parent class.
        super().__init__(model)
        self.cell = cell

        # Create the agent's variable and set the initial values.
        self.performanceMeasure = 0

        self.LLMstring = ""
        self.turn = 3
        self.dead = False        # iv. has a flag dead

        self.isWumpus = False    # ii. has a flag isWumpus
        self.isStench = False
        self.senseStench = False
        self.shots_left = 1     # iii. has a variable shots_left = 15

        self.isPit = False
        self.isBreeze = False
        self.senseBreeze = False

        self.isGold = False
        self.isGlitter = False
        self.senseGlitter = False
        self.goldGrabbed = False
        self.inventory = "None"
        self.goldReleased = False

        self.performanceMeasure = 0
        
    def step(self):
        self.move()                                         # ACTUATORS : LLM SHOULD CONTROL THIS
        self.place_stench()
        self.wumpus_world()                                 # SENSORS update
        if(not self.isWumpus and not self.dead):            
            print("BGS LLM string: " + self.LLMstring)      # This is the "NotImplementedHeroAgent"
        #if(not self.isWumpus):                             #Performance measure: uncomment to view
        #    print("Performance measure: " + str(self.performanceMeasure))

    #ACTUATORS
    #this selects to go turn either: left, right, or go forward randomly. Note: an agent does not move forward every turn
    def move(self):
        self.glitter_response()  #put in move, the hero must choose to pick it up
        self.return_home()       #put in move
        movementChoice = random.randint(0,2) 
        if(not self.isWumpus and not self.dead):
            print("Hero movement choice: " + str(movementChoice))
            if (movementChoice == 0):
                self.turn_left()
                self.performanceMeasure -=1
            elif (movementChoice == 1):
                self.turn_right()
                self.performanceMeasure -=1
            else:
                self.forward()
                self.performanceMeasure -=1
        else:
            if (movementChoice == 0):
                self.turn_left()
            elif (movementChoice == 1):
                self.turn_right()
            else:
                self.forward()

    def turn_left(self):
        if not self.dead:
            self.turn+=1
            if not self.isWumpus:
                print(f"Hero {self.model.agents.index(self)} turned left.")
            #else:
            #    print(f"Wumpus {self.model.agents.index(self)} turned left.")  

    def turn_right(self):
        if not self.dead:
            self.turn-=1
            if not self.isWumpus:
                print(f"Hero {self.model.agents.index(self)} turned right.")
            #else:
            #    print(f"Wumpus {self.model.agents.index(self)} turned right.")

    def forward(self):
        if not self.dead:
            x, y = self.pos
            if (self.turn%4==0):
                new_position = (x - 1, y)
                if not self.isWumpus:
                    print(f"Hero {self.model.agents.index(self)} went left.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went left.")     
            if (self.turn%4==1):
                new_position = (x, y - 1)
                if not self.isWumpus:
                    print(f"Hero {self.model.agents.index(self)} went down.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went down.")
            if (self.turn%4==2):
                new_position = (x + 1, y)
                if not self.isWumpus:
                    print(f"Hero {self.model.agents.index(self)} went right.")
                #else:
                #    print(f"Wumpus {self.model.agents.index(self)} went right.") 
            if (self.turn%4==3):
                new_position = (x, y + 1)
                if not self.isWumpus:
                    print(f"Hero {self.model.agents.index(self)} went up.")
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
                #print(f"This gold {self.model.agents.index(self.inventory)} of {self.inventory} has been grabbed")
                print(f"This gold {self.inventory} has been grabbed")
                if gold_agent:
                    gold_agent = gold_agent[0]
                    self.model.grid.remove_agent(gold_agent)
                    print("Gold has been removed from the grid.")

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
        self.LLMstring = ""
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
                self.LLMstring += "Hero senses a stench. "
            elif (self.senseBreeze):
                self.LLMstring += "Hero senses a breeze. "
            elif (self.senseGlitter):
                self.LLMstring += "Hero senses a glitter. "
        if(not self.isWumpus and not self.dead):
            print("LLM string: " + self.LLMstring)
            return self.LLMstring
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
    name="Wumpus Outbreak Model"
)

# This is required to render the visualization in the Jupyter notebook
page