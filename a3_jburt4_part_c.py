#To run: solara run a3_jburt4_part_c.py

import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

#ZombieHumanPlot two lines on make_plot_component (top-right plot)
def compute_zombies(model):
    agent_zombies = int(sum(agent.isZombie and not agent.dead for agent in model.agents))
    #print(f"Zombies: {agent_zombies}")
    return agent_zombies
def compute_humans(model):
    agent_humans = int(sum(not agent.isZombie and not agent.dead for agent in model.agents))
    #print(f"Humans: {agent_humans}")
    return agent_humans

class OutbreakAgent(mesa.Agent):
    """An agent with a zombie outbreak."""

    def __init__(self, model):
        super().__init__(model)

        self.isZombie = False
        self.shots_left = 15
        self.dead = False
        self.step_counter = 0 
        self.previous_step = -1
        self.num_cell_humans = 0
        self.human_cellmate_ids = []

        self.initialPrompt = "Your name is Agent " + str(self.model.agents.index(self)) + ", a friendly townsperson in a simulation. Choices: If zombies or out of ammo, go towards (0,0) = Choice 0. If at (0,0), move one square out diagonally = Choice 1. The last thing you write is your choice. "
        self.SENSORstring = ""                                                   # updated in outbreak_area()
        self.parsed_response = None
        self.MEMORYstring = ["Find a way to survive from the zombies. "]           # updated in move() 
        self.MOVEMENTstring = []

    def step(self):
        self.step_counter += 1
        self.move()              
        self.outbreak_area()                                # If there is 2 or more agents in a cell, only 1 SENSORYstring will show because the other agents move out of the cell during that step
        #NOTE: A conversation between agents would have to occur after ...?
 

    def zombies_per_cell(self,cellmates):
        if(not self.isZombie and not self.dead):
            self.num_cell_zombies = 0
            for i in cellmates:
                if i.isZombie == True:
                    self.num_cell_zombies +=1
            return self.num_cell_zombies    

    def humans_per_cell(self,cellmates):
        if(not self.isZombie and not self.dead):
            self.num_cell_humans = 0
            for i in cellmates:
                if i.isZombie == False:
                    self.num_cell_humans +=1    
            return self.num_cell_humans    
    
    def human_ids_in_cell(self,cellmates):
        if(not self.isZombie and not self.dead):
            self.human_cellmate_ids = []
            for i in cellmates:
                if(not i.isZombie and i.model.agents.index(i) != self.model.agents.index(self)):
                    self.human_cellmate_ids.append(i.model.agents.index(i))
                    #HUMAN IN SAME CELL COMMUNICATION CONFIRMATION:
                    #print("IDs in the same cell:")
                    #print(self.model.agents.index(self))
                    #print(i.model.agents.index(i))
                    #print(self.human_cellmate_ids)
            return self.human_cellmate_ids  
        
    def move(self): 
        if(self.isZombie and not self.dead):
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position) #this will move to a random adjacent grid square (1 square per render interval)

        if(not self.isZombie and not self.dead):
            #HUMAN IN SAME CELL COMMUNICATION CONFIRMATION:
            #if(self.num_cell_humans > 1):
            #    cellmates = self.model.grid.get_cell_list_contents([self.pos])
            #    for i in cellmates:
            #        if(not i.isZombie):
            #            print(i.pos)
            #            print(i.model.agents.index(i))

            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)
            
            # 1. LLM DECISION: IF 0, GO TOWARDS (0,0). IF 1, GO TO A RANDOM CELL
            #if(len(self.MOVEMENTstring) > 0):
            #    print("Agent " + str(self.model.agents.index(self))+ " choice: " + str(self.MOVEMENTstring[-1]) + " of all choices made: " + str(self.MOVEMENTstring[::-1]) + " at simulation step "  + str(self.step_counter))

            if(len(self.MOVEMENTstring) > 0 and (self.MOVEMENTstring[-1]) == 0 and self.pos != (0,0)): # Choice 0 and not at (0,0)
                if self.pos[0] != 0:
                    new_x = self.pos[0] - 1
                else:
                    new_x = self.pos[0]
                if self.pos[1] != 0:
                    new_y = self.pos[1] - 1
                else:
                    new_y = self.pos[1]
                new_position = (new_x, new_y)
            elif(len(self.MOVEMENTstring) > 0 and (self.MOVEMENTstring[-1]) == 0 and self.pos == (0,0)): # Choice 0 and at (0,0)
                new_position = self.pos
            elif(len(self.MOVEMENTstring) > 0 and (self.MOVEMENTstring[-1]) == 1):                       # Choice 1 
                new_x = self.pos[0] + 1
                new_y = self.pos[1] + 1
                new_position = (new_x, new_y)
            else:
                new_position = self.random.choice(possible_steps)                                        # Choice random
            self.model.grid.move_agent(self, new_position) #this will move to an adjacent grid square (1 square per render interval)


    #self.SENSORstring AND IMMEDIATE ACTIONS TAKE PLACE HERE
    def outbreak_area(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        self.zombies_per_cell(cellmates)
        self.num_cell_humans = self.humans_per_cell(cellmates)
        self.human_ids_in_cell(cellmates)
        self.human_cellmate_ids = self.human_ids_in_cell(cellmates)

        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            
            # Simulation step:
            # ZOMBIE CHOICE:            
            PercentChanceZ = random.randint(0, 1)       
            PercentChanceZ_ammo = random.randint(0, 1)    
            if(self.isZombie and not self.dead and not other.isZombie and not other.dead and PercentChanceZ < 0.5):             
                other.isZombie = True
                #print(f"Human {self.model.agents.index(other)} is now a zombie.")   
                return       
            if(self.isZombie and not other.isZombie and not other.dead and self.shots_left > 0 and PercentChanceZ >= 0.5 and PercentChanceZ_ammo < 0.5):
                self.shots_left -=1
                #print(f"Zombie {self.model.agents.index(self)} dropped 1 ammo, remaining: {self.shots_left}")
                other.shots_left +=1
                #print(f"Human {self.model.agents.index(other)} picked up ammo. Shots left: {other.shots_left}")    
                return      
                                        
            # HUMAN AGENT CHOICE:
            PercentChanceH = random.randint(0, 1)                    
            if(not self.isZombie and not self.dead and other.isZombie and PercentChanceH < 0.5 and self.shots_left > 0): 
                self.shots_left -= 1
                other.dead = True   
                other.isZombie = False
                other.shots_left = 0  
                if(self.num_cell_humans > 1):

                    # SIMULATION STEP: SENSORS 
                    print("\nSIMULATION STEP: " + str(self.step_counter))
                    self.SENSORstring = "This is agent " + str(self.model.agents.index(self)) + " with agent(s) " + str(self.human_cellmate_ids) + ". There are " + str(self.zombies_per_cell(cellmates)) + " zombies in cell " + str(self.pos) + ". Zombie " + str(self.model.agents.index(other)) + " down. Shots left: " + str(self.shots_left) + ". "
                    print(f"SENSORS (ZOMBIE DOWN): {self.SENSORstring}") 
                    self.MEMORYstring.append(self.SENSORstring)     #NOTE: this may not be used as the PromptModel memorystream because of the immediate situation 

                    #CONVERSATION
                    print("THE CONVERSATION: ")
                    conversation_initiator = self.model.PromptModel(self.initialPrompt, str(self.MEMORYstring), "Agent " + str(self.model.agents.index(self)) + " in contact with " + str(self.human_cellmate_ids) + ", my thoughts are a maximum of 30 words. ")
                    print(f"INITIAL CONTACT (ZOMBIE DOWN): {conversation_initiator}") 
                    self.MEMORYstring.append(conversation_initiator)
                    ### CURRENT AGENT CHOICE
                    reversed_response = conversation_initiator[::-1]
                    for char in reversed_response:
                        if char.isdigit():
                            self.parsed_response = int(char)
                            break
                        else:
                            self.parsed_response = 2
                    self.MOVEMENTstring.append(self.parsed_response)
                    ###
                    conversation = [""] * len(self.model.agents)
                    intial_context = conversation_initiator + "Choices: Go to safety at (0,0) = 0. Expand territory = 1. The last thing you write is your choice. "
                    agent_response = ""
                    for i in cellmates:
                        if(not self.isZombie and not self.dead and not i.isZombie and not i.dead and i.model.agents.index(i) != self.model.agents.index(self)):
                            conversation[i.model.agents.index(i)] = "I am agent "+ str(i.model.agents.index(i)) + ". Looks someone wants to talk me. "
                            agent_response = self.model.PromptModel(intial_context, conversation[i.model.agents.index(i)], "Give your name and respond in a different way, 30 words max. Give your choice at the end of the conversation.")
                            print(f"RESPONSE: {agent_response}")
                            ### CONTACTED AGENT CHOICE
                            reversed_response = agent_response[::-1]
                            for char in reversed_response:
                                if char.isdigit():
                                    i.parsed_response = int(char)
                                    break
                                else:
                                    i.parsed_response = 2
                            i.MOVEMENTstring.append(i.parsed_response)
                            ###
                            conversation[i.model.agents.index(i)] += agent_response
                            i.MEMORYstring.append(str(conversation[i.model.agents.index(i)]))  # save conversation to i
                            self.MEMORYstring.append(agent_response)                           # save conversation to self
                    # ALL MEMORIES OF CURRENT AGENT SINCE START
                            #print("AGENT " + str(i.model.agents.index(i)) + "'S TOTAL MEMORIES: " + str(i.MEMORYstring))
                    #print("AGENT " + str(self.model.agents.index(self)) + "'S TOTAL MEMORIES: " + str(self.MEMORYstring))

            #missed shot
            if(not self.isZombie and not self.dead and other.isZombie and PercentChanceH >= 0.5 and self.shots_left > 0): 
                self.shots_left -= 1
                if(self.num_cell_humans > 1):

                    # SIMULATION STEP: SENSORS 
                    print("\nSIMULATION STEP: " + str(self.step_counter))
                    self.SENSORstring = "I am agent " + str(self.model.agents.index(self)) + " with agent(s) " + str(self.human_cellmate_ids) + ". There are " + str(self.zombies_per_cell(cellmates)) + " zombies in cell " + str(self.pos) + ". Missed the shot. Shots left: " + str(self.shots_left) + ". "
                    print(f"SENSORS (MISSED SHOT): {self.SENSORstring}") 
                    self.MEMORYstring.append(self.SENSORstring)     #NOTE: this may not be used as the PromptModel memorystream because of the immediate situation 

                    #CONVERSATION
                    print("THE CONVERSATION: ")
                    conversation_initiator = self.model.PromptModel(self.initialPrompt, str(self.MEMORYstring), "Agent " + str(self.model.agents.index(self)) + " in contact with " + str(self.human_cellmate_ids) + ", my thoughts are a maximum of 30 words. ")
                    print(f"INITIAL CONTACT (MISSED SHOT): {conversation_initiator}") 
                    self.MEMORYstring.append(conversation_initiator)
                    ### CURRENT AGENT CHOICE
                    reversed_response = conversation_initiator[::-1]
                    for char in reversed_response:
                        if char.isdigit():
                            self.parsed_response = int(char)
                            break
                        else:
                            self.parsed_response = 2
                    self.MOVEMENTstring.append(self.parsed_response)
                    ###
                    conversation = [""] * len(self.model.agents)
                    intial_context = conversation_initiator + "Choices: Go to safety at (0,0) = 0. Expand territory = 1. The last thing you write is your choice. "
                    agent_response = ""
                    for i in cellmates:
                        if(not self.isZombie and not self.dead and not i.isZombie and not i.dead and i.model.agents.index(i) != self.model.agents.index(self)):
                            conversation[i.model.agents.index(i)] = "I am agent "+ str(i.model.agents.index(i)) + ". Looks someone wants to talk me. "
                            agent_response = self.model.PromptModel(intial_context, conversation[i.model.agents.index(i)], "Give your name and respond in a different way, 30 words max. Give your choice at the end of the conversation.")
                            print(f"RESPONSE: {agent_response}")
                            ### CONTACTED AGENT CHOICE
                            reversed_response = agent_response[::-1]
                            for char in reversed_response:
                                if char.isdigit():
                                    i.parsed_response = int(char)
                                    break
                                else:
                                    i.parsed_response = 2
                            i.MOVEMENTstring.append(i.parsed_response)
                            ###
                            conversation[i.model.agents.index(i)] += agent_response
                            i.MEMORYstring.append(str(conversation[i.model.agents.index(i)]))  # save conversation to i
                            self.MEMORYstring.append(agent_response)                           # save conversation to self
                    # ALL MEMORIES OF CURRENT AGENT SINCE START
                            #print("AGENT " + str(i.model.agents.index(i)) + "'S TOTAL MEMORIES: " + str(i.MEMORYstring))
                    #print("AGENT " + str(self.model.agents.index(self)) + "'S TOTAL MEMORIES: " + str(self.MEMORYstring))

            #run out of ammo
            if(not self.isZombie and not self.dead and other.isZombie and self.shots_left == 0): 
                if(self.num_cell_humans > 1):
                    
                    # SIMULATION STEP: SENSORS 
                    print("\nSIMULATION STEP: " + str(self.step_counter))
                    self.SENSORstring = "I am agent " + str(self.model.agents.index(self)) + " with agent(s) " + str(self.human_cellmate_ids) + ". There are " + str(self.zombies_per_cell(cellmates)) + " zombies in cell " + str(self.pos) + ". Out of ammo. Shots left: " + str(self.shots_left) + ". Request immediate assistance. "
                    print(f"SENSORS (NO AMMO): {self.SENSORstring}") 
                    self.MEMORYstring.append(self.SENSORstring)     #NOTE: this may not be used as the PromptModel memorystream because of the immediate situation 

                    #CONVERSATION
                    print("THE CONVERSATION: ")
                    conversation_initiator = self.model.PromptModel(self.initialPrompt, str(self.MEMORYstring), "Agent " + str(self.model.agents.index(self)) + " in contact with " + str(self.human_cellmate_ids) + ", my thoughts are a maximum of 30 words. ")
                    print(f"INITIAL CONTACT (NO AMMO): {conversation_initiator}") 
                    self.MEMORYstring.append(conversation_initiator)
                    ### CURRENT AGENT CHOICE
                    reversed_response = conversation_initiator[::-1]
                    for char in reversed_response:
                        if char.isdigit():
                            self.parsed_response = int(char)
                            break
                        else:
                            self.parsed_response = 2
                    self.MOVEMENTstring.append(self.parsed_response)
                    ###
                    conversation = [""] * len(self.model.agents)
                    intial_context = conversation_initiator + "Choices: Go to safety at (0,0) = 0. Expand territory = 1. The last thing you write is your choice. "
                    agent_response = ""
                    for i in cellmates:
                        if(not self.isZombie and not self.dead and not i.isZombie and not i.dead and i.model.agents.index(i) != self.model.agents.index(self)):
                            conversation[i.model.agents.index(i)] = "I am agent "+ str(i.model.agents.index(i)) + ". Looks someone wants to talk me. "
                            agent_response = self.model.PromptModel(intial_context, conversation[i.model.agents.index(i)], "Give your name and respond in a different way, 30 words max. Give your choice at the end of the conversation.")
                            print(f"RESPONSE: {agent_response}")
                            ### CONTACTED AGENT CHOICE
                            reversed_response = agent_response[::-1]
                            for char in reversed_response:
                                if char.isdigit():
                                    i.parsed_response = int(char)
                                    break
                                else:
                                    i.parsed_response = 2
                            i.MOVEMENTstring.append(i.parsed_response)
                            ###
                            conversation[i.model.agents.index(i)] += agent_response
                            i.MEMORYstring.append(str(conversation[i.model.agents.index(i)]))  # save conversation to i
                            self.MEMORYstring.append(agent_response)                           # save conversation to self
                    # ALL MEMORIES OF CURRENT AGENT SINCE START
                            #print("AGENT " + str(i.model.agents.index(i)) + "'S TOTAL MEMORIES: " + str(i.MEMORYstring))
                    #print("AGENT " + str(self.model.agents.index(self)) + "'S TOTAL MEMORIES: " + str(self.MEMORYstring))
            
            #two humans in a cell without zombies
            if(not self.isZombie and not self.dead and not other.isZombie): 
                if(self.num_cell_humans > 1):
                    # SIMULATION STEP: SENSORS 
                    print("\nSIMULATION STEP: " + str(self.step_counter))
                    self.SENSORstring = "I am agent " + str(self.model.agents.index(self)) + " with agent(s) " + str(self.human_cellmate_ids) + ". There are " + str(self.zombies_per_cell(cellmates)) + " zombies in cell " + str(self.pos) + ". "
                    print(f"SENSORS (ONLY HUMANS): {self.SENSORstring}") 
                    self.MEMORYstring.append(self.SENSORstring)     #NOTE: this may not be used as the PromptModel memorystream because of the immediate situation 

                    #CONVERSATION
                    print("THE CONVERSATION: ")
                    conversation_initiator = self.model.PromptModel(self.initialPrompt, str(self.MEMORYstring), "Agent " + str(self.model.agents.index(self)) + " in contact with " + str(self.human_cellmate_ids) + ", my thoughts are a maximum of 30 words. ")
                    print(f"INITIAL CONTACT (ONLY HUMANS): {conversation_initiator}") 
                    self.MEMORYstring.append(conversation_initiator)
                    ### CURRENT AGENT CHOICE
                    reversed_response = conversation_initiator[::-1]
                    for char in reversed_response:
                        if char.isdigit():
                            self.parsed_response = int(char)
                            break
                        else:
                            self.parsed_response = 2
                    self.MOVEMENTstring.append(self.parsed_response)
                    ###
                    conversation = [""] * len(self.model.agents)
                    intial_context = conversation_initiator + "Choices: Go to safety at (0,0) = 0. Expand territory = 1. The last thing you write is your choice. "
                    agent_response = ""
                    for i in cellmates:
                        if(not self.isZombie and not self.dead and not i.isZombie and not i.dead and i.model.agents.index(i) != self.model.agents.index(self)):
                            conversation[i.model.agents.index(i)] = "I am agent "+ str(i.model.agents.index(i)) + ". Looks someone wants to talk me. "
                            agent_response = self.model.PromptModel(intial_context, conversation[i.model.agents.index(i)], "Give your name and respond in a different way, 30 words max. Give your choice at the end of the conversation.")
                            print(f"RESPONSE: {agent_response}")
                            ### CONTACTED AGENT CHOICE
                            reversed_response = agent_response[::-1]
                            for char in reversed_response:
                                if char.isdigit():
                                    i.parsed_response = int(char)
                                    break
                                else:
                                    i.parsed_response = 2
                            i.MOVEMENTstring.append(i.parsed_response)
                            ###
                            conversation[i.model.agents.index(i)] += agent_response
                            i.MEMORYstring.append(str(conversation[i.model.agents.index(i)]))  # save conversation to i
                            self.MEMORYstring.append(agent_response)                           # save conversation to self
                    # ALL MEMORIES OF CURRENT AGENT SINCE START
                            #print("AGENT " + str(i.model.agents.index(i)) + "'S TOTAL MEMORIES: " + str(i.MEMORYstring))
                    #print("AGENT " + str(self.model.agents.index(self)) + "'S TOTAL MEMORIES: " + str(self.MEMORYstring))
                    

class OutbreakModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, totalAgents=100, percentZombie=10, width=20, height=20): # CHANGED : percentZombie=10
        super().__init__()
        self.total_agents = totalAgents
        self.percent_zombie = percentZombie # CHANGED
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Zombies": compute_zombies, "Humans": compute_humans},agent_reporters={"Wealth": "wealth"}
        )
        # Create agents 
        for i in range(self.total_agents):
            agent = OutbreakAgent(self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        self.turn_into_zombie() # CHANGED

        self.running = True
        #self.datacollector.collect(self)

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

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
    
    def turn_into_zombie(self):     
        num_zombies = int(self.total_agents *(self.percent_zombie/100))    
        print(f"Total zombie agents: {num_zombies}")    
        for i in range(num_zombies):                    
            self.agents[i].isZombie = True              
        print(f"Total regular agents: {self.total_agents-num_zombies}")    


model_params = {
    "totalAgents": {
        "type": "SliderInt",
        "value": 100,   # matched to initial settings, this can change on user RESET
        "label": "Number of agents:",
        "min": 1,
        "max": 100,
        "step": 1,
    },
    #UPGRADE 2: % Zombie slider: this changes the percentage of agents as zombies on the grid after sliding the bar and pressing reset.
    "percentZombie": {
        "type": "SliderInt", 
        "value": 10,    # matched to initial settings, this can change on user RESET
        "label": "% Zombie:", 
       "min": 1,    
        "max": 100,  
        "step": 1,  
    },
    "width": {
        "type": "SliderInt",
        "value": 20,    # matched to initial settings, this can change on user RESET
        "label": "Width:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
    "height": {
        "type": "SliderInt",
        "value": 20,    # matched to initial settings, this can change on user RESET
        "label": "Height:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
}

#modify this function to change output on grid
def agent_portrayal(agent):
    size = 10
    color = "tab:red"

    if agent.isZombie: 
        size = 60  
        color = "tab:green"    
    
    if agent.dead:
        size = 100 
        color = "tab:orange"    

    return {"size": size, "color": color}

outbreak_model = OutbreakModel(20, 20, 10, 10) # v. When you create your agents, set 10% of them to be Zombies

def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


SpaceGraph = make_space_component(agent_portrayal)

def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))

ZombieHumanPlot = make_plot_component(
    {"Zombies": "tab:green", "Humans": "tab:red"},
    post_process=post_process_lines
)

page = SolaraViz(
    outbreak_model,
    components=[SpaceGraph, ZombieHumanPlot],
    model_params=model_params,
    name="Zombie Outbreak Model"
)

# This is required to render the visualization in the Jupyter notebook
page
