import random
import timeit
import time as t
import numpy as np
import copy
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import seaborn as sns

class grid_world_solver:
    def __init__(self,world,reward,gamma,time,transition_model):
        self.megatable = {}
        self.world = world
        self.decay_rate = 0.99
        self.reward = reward
        self.gamma = gamma
        self.time = time
        self.transition_model = transition_model
        self.row = len(self.world)
        self.col = len(self.world[0])
        self.alpha = 0.1
        self.policymap = np.zeros((self.row, self.col), dtype='object')
        self.heatmap = np.zeros((self.row, self.col), dtype='object')
        self.e = 0.9
        self.nomove = []
        self.win_pos = []
        self.loose_pos = []
        self.loose_reward = []
        self.win_reward = []
        self.totreward = 0
        self.hole = []
        self.hole_letter = []

        self.meanreward = []
        self.time_reward = []

        for i in range(self.row):
            for j in range(self.col):
                if self.world[i][j] == 'S':
                    self.start = (i, j)
                elif self.world[i][j] == 'X':
                    self.nomove.append((i, j))
                elif not type(self.world[i][j]) == int and not type(self.world[i][j]) == float and self.world[i][j].isalpha() == True and self.world[i][j] not in ['X', 'S']:
                    self.hole.append((i, j))
                    self.hole_letter.append(self.world[i][j])
                elif self.world[i][j] < 0:
                    self.loose_pos.append((i, j))
                    self.loose_reward.append(self.world[i][j])
                elif self.world[i][j] > 0:
                    self.win_pos.append((i, j))
                    self.win_reward.append(self.world[i][j])
        
    def searchworld(self,x):
        flag=0
        loc=[]
        for i in range(self.row):
            for j in range(self.col):
                if self.world[i][j]==x:
                    flag=1
                    loc=[i,j]
            if flag==1:
                break
        return loc

    def print2D(self,m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                print(m[i][j], end="\t")
            print()

    def printdict(self,d):
        for i in d:
            print(i,':',d[i])

    def rsttable(self):
        for i in range(self.row):
            for j in range(self.col):
                if self.world[i][j]!='X':
                    self.megatable.update({(i,j):[[0,0,0,0],[0,0,0,0]]})

    def getsquares(self,square):
        coords=[]
        if (square[0]-1)>=0 and self.world[square[0]-1][square[1]]!='X':
            coords.append([(square[0]-1),square[1]])
        else:
            coords.append([(square[0]),square[1]])
        if (square[0]+1)<self.row and self.world[square[0]+1][square[1]]!='X':
            coords.append([(square[0]+1),square[1]])
        else:
            coords.append([(square[0]),square[1]])
        if (square[1]-1)>=0 and self.world[square[0]][square[1]-1]!='X':
            coords.append([(square[0]),square[1]-1])
        else:
            coords.append([(square[0]),square[1]])
        if (square[1]+1)<self.col and self.world[square[0]][square[1]+1]!='X':
            coords.append([(square[0]),square[1]+1])
        else:
            coords.append([(square[0]),square[1]])
        return coords

    def vcheck(self,s): #Checks whether a square is valid 
        if s[0]<self.row and s[0]>=0 and s[1]<self.col and s[1]>=0 and self.world[s[0]][s[1]]!='X':
            return True 
        else:
            return False

    def makemove(self, csquare, nsquare): #accepts 2 valid squares and returns the square after the move
        flag=0
        if nsquare[0]-csquare[0]!=0:
            if nsquare[0]-csquare[0]>0:
                flag=2 #down
            else:
                flag=1 #up
        elif nsquare[1]-csquare[1]!=0:
            if nsquare[1]-csquare[1]>0:
                flag=4 #right
            else:
                flag=3 #left
        else:
            return csquare

        rand=random.random()
        
        if rand<=self.transition_model:
            return nsquare
        else:
            rand=random.random()
            if rand>=0.5:
                if flag==1 and grid_world_solver.vcheck(self,[nsquare[0]-1,nsquare[1]]):
                    nsquare[0]=nsquare[0]-1
                elif flag==2 and grid_world_solver.vcheck(self,[nsquare[0]+1,nsquare[1]]):
                    nsquare[0]=nsquare[0]+1
                elif flag==3 and grid_world_solver.vcheck(self,[nsquare[0],nsquare[1]-1]):
                    nsquare[1]=nsquare[1]-1
                elif flag==4 and grid_world_solver.vcheck(self,[nsquare[0],nsquare[1]+1]):
                    nsquare[1]=nsquare[1]+1
            else:
                if flag==1:
                    if grid_world_solver.vcheck(self,[nsquare[0]+2,nsquare[1]]):
                        nsquare[0]=nsquare[0]+2
                    else:
                        nsquare=csquare
                elif flag==2:
                    if grid_world_solver.vcheck(self,[nsquare[0]-2,nsquare[1]]):
                        nsquare[0]=nsquare[0]-2
                    else:
                        nsquare=csquare
                elif flag==3:
                    if grid_world_solver.vcheck(self,[nsquare[0],nsquare[1]+2]):
                        nsquare[1]=nsquare[1]+2
                    else:
                        nsquare=csquare
                elif flag==4:
                    if grid_world_solver.vcheck(self,[nsquare[0],nsquare[1]-2]):
                        nsquare[1]=nsquare[1]-2
                    else:
                        nsquare=csquare
            return nsquare

    def updatetable(self, s, a, r, sdash, adash):
        self.megatable[tuple(s)][0][a]=self.megatable[tuple(s)][0][a]+self.alpha*(r+self.reward+self.gamma*self.megatable[tuple(sdash)][0][adash]-self.megatable[tuple(s)][0][a])
        self.megatable[tuple(s)][1][a]+=1
        
    def policyeg(self, square, e): #epsilon-greedy policy
        rand=random.random()
        actions=self.megatable[(square[0],square[1])][0]
        if rand>e:
            m=max(actions)
            alist=[i for i,x in enumerate(actions) if x==m]
            action=alist[random.randint(0,(len(alist)-1))]
        else:
            action= random.randint(0,3)
        return action

    def sarsa(self):

        count = 0
        self.e = 0.9
        grid_world_solver.rsttable(self)
        start=grid_world_solver.searchworld(self,'S')
        s=start
        t0=t.time()
        a=grid_world_solver.policyeg(self,s,self.e)

        while t.time()-t0<self.time:
            s_dash=grid_world_solver.makemove(self,s,grid_world_solver.getsquares(self,s)[a])
            # a_dash=grid_world_solver.policyeg(self,s_dash,self.e)
            
            r=self.world[s_dash[0]][s_dash[1]]
            if type(r) != int and type(r) != float:
                r=0

            #Check if it's a terminal state
            if s_dash in self.hole:
                ind = self.hole.index(s_dash)
                letter = self.hole_letter[ind]
                for wormhole in self.hole:
                    if self.world[wormhole[0]][wormhole[1]] == letter and wormhole != s_dash:
                        s_dash = wormhole
            a_dash=grid_world_solver.policyeg(self,s_dash,self.e)
            grid_world_solver.updatetable(self, s, a, r, s_dash, a_dash)
            self.totreward += (r + self.reward)

            if self.world[s_dash[0]][s_dash[1]] != 0 and (type(self.world[s_dash[0]][s_dash[1]]) == int or type(self.world[s_dash[0]][s_dash[1]]) == float):
                s_dash = start.copy()
                count += 1

                self.meanreward.append(self.totreward/count)
                self.time_reward.append(t.time()-t0)
            s=s_dash.copy()
            a=a_dash

        self.genheatmap()
        self.genpolicymap()
        self.plot()
        self.plotreward()
        #e = e*self.decay_rate

    def genpolicymap(self):
        for i in range(self.row):
            for j in range(self.col):
                if self.world[i][j]!='X' and self.world[i][j]==0:
                    a = self.megatable[(i,j)][0].index(max(self.megatable[(i,j)][0]))
                    if a==0:
                        self.policymap[i][j]='↑'
                    elif a==1:
                        self.policymap[i][j]='↓'
                    elif a==2:
                        self.policymap[i][j]='←'
                    elif a==3:
                        self.policymap[i][j]='→'
                else:
                    self.policymap[i][j] = self.world[i][j]

        print("Policy map: ")
        print(self.policymap)

    def genheatmap(self):
        s=0
        for i in range(self.row):
            for j in range(self.col):
                if self.world[i][j]!='X': 
                    a=sum(self.megatable[(i,j)][1])
                    s=s+a
        for i in range(self.row):
            for j in range(self.col):
                if self.world[i][j]!='X': 
                    a =sum(self.megatable[(i,j)][1])
                    self.heatmap[i][j]=round(100*a/s,3)
        
        print("Heat map: ")
        print(self.heatmap)

        f = plt.figure(1)
        #plot heatmap
        heatmap_df = pd.DataFrame(self.heatmap)
        heatmap_df = heatmap_df[heatmap_df.columns].astype(float)
        #heatmap_df = [heatmap_df[i].astype(float) for i in range(len(heatmap_df))]
        color = sns.color_palette("Reds")        
        ax = sns.heatmap(heatmap_df, annot=True, cmap=color, linewidth=0.5)
        plt.title('Heat map', fontweight="bold")
        f.show()

    def plotreward(self):
        fig = plt.figure()
        plt.plot(self.time_reward, self.meanreward)
        print("Average reward is:", round(sum(self.meanreward)/len(self.meanreward),4))
        plt.xlabel('Time (s)', fontweight="bold")
        plt.ylabel('Mean Reward', fontweight="bold")
        plt.show()

    def plot(self):

        #plot heat map
        fig, ax = plt.subplots(1)
        ax.margins()

        for i in range(self.row):
            for j in range(self.col):
                if (i, j) in self.nomove:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='k')) 
                elif (i, j) == self.start:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='r'))
                    plt.text(j, i, 'S', fontsize=100/(max(self.row, self.col)), ha='center', va='center')
                elif self.world[i][j] != 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='g'))
                    plt.annotate(str(self.world[i][j]), fontsize=100/(max(self.row, self.col)), ha='center', va='center', xytext=(j, i-0.1), xy=(j, i-0.1))
                else:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='w'))
                    plt.text(j, i, self.policymap[i][j], fontsize=150/(max(self.row, self.col)), ha='center', va='center', color='black')

        g = plt.figure(2)
        plt.title('Policy map', fontweight="bold")
        plt.axis('scaled')
        plt.gca().invert_yaxis()
        g.show()
        
        fig, ax = plt.subplots(1)
        ax.margins()
        for i in range(self.row):
            for j in range(self.col):
                if (i, j) in self.nomove:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1,
                                 1, edgecolor='k', facecolor='k'))
                elif (i, j) == self.start:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1,
                                 1, edgecolor='k', facecolor='r'))
                    plt.text(j, i, 'S', fontsize=100 /
                             (max(self.row, self.col)), ha='center', va='center')
                elif self.world[i][j] != 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1,
                                 1, edgecolor='k', facecolor='g'))
                    plt.annotate(str(self.world[i][j]), fontsize=100/(max(
                        self.row, self.col)), ha='center', va='center', xytext=(j, i-0.1), xy=(j, i-0.1))
                else:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1,
                                 1, edgecolor='k', facecolor='w'))
                    plt.text(j, i, self.heatmap[i][j], fontsize=100/(
                        max(self.row, self.col)), ha='center', va='center', color='black')
        g_ = plt.figure(3)
        plt.title('Heat map', fontweight="bold")
        plt.axis('scaled')
        plt.gca().invert_yaxis()
        g_.show()
    
def getworld(file):

    df = pd.read_csv(file, sep='\t', header=None)
    dataset = pd.DataFrame(df)
    grid = np.array(dataset)
    rows = grid.shape[0]
    col = grid.shape[1]

    for i in range(rows):
        for j in range(col):
            if not type(grid[i][j]) == int and not type(grid[i][j]) == float and not grid[i][j].isalpha():
                grid[i][j] = int(grid[i][j])
    
    w = np.ndarray.tolist(grid)

    return w

if __name__ == "__main__":
    #set default values
    default_gamma = 0.99
    default_time = 20
    default_source = 'gridworld_wormhole.txt'
    default_reward = -0.1
    default_probability = 0.75

    file = sys.argv[1]
    #file = input("What's the input file for the gridworld? ") or default_source
    #type(file)

    world = getworld(file)

    reward = sys.argv[2]
    #reward = input("Input the reward: ") or default_reward
    #type(reward)

    gamma = sys.argv[3]
    #gamma = input("Input the discount parameter: ") or default_gamma
    #type(gamma)

    time = sys.argv[4]
    #time = input("Input the desired running time: ") or default_time
    #type(time)

    transition_model = sys.argv[5]
    #transition_model = input("Input the transition model: ") or default_probability
    #type(transition_model) 
   
    gws = grid_world_solver(world, float(reward), float(gamma), float(time), float(transition_model))
    gws.sarsa()