import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
pi = math.pi
sqrt = math.sqrt

random.seed(123)
coords = [(round(random.uniform(0,1),5), round(random.uniform(0,1),5)) for _ in range(1200)]
coords = pd.DataFrame(coords, columns = ['x', 'y'])


def sort_coordinates(x,y):
    if sqrt((x-(1-sqrt(0.3/pi)))**2+(y-(1-sqrt(0.3/pi)))**2) <= sqrt(0.3/pi): # 30% area circle
        return 'B'
    elif sqrt((x-sqrt(0.2/pi))**2+(y-sqrt(0.2/pi))**2) <= sqrt(0.2/pi): 
        return 'B'
    elif sqrt((x-(1-sqrt(0.1/pi)))**2+(y-sqrt(0.1/pi))**2) <= sqrt(0.1/pi): # 10% area circle
        return 'B'
    else:
        return 'A'
    
coords['class'] = coords.apply(lambda x: sort_coordinates(x.x, x.y), axis=1)

coordsA = coords[coords['class'] == 'A']
coordsB = coords[coords['class'] == 'B']

coordsA = coords[coords['class'] == 'A'].sample(300, replace=False)
coordsB = coords[coords['class'] == 'B'].sample(300, replace=False)



#plt.plot(coords['x'], coords['y'], '.', )

#plt.show()

fig, ax = plt.subplots()

colors = {'A':'red', 'B':'green'}


ax.scatter(coords['x'], coords['y'], c=coords['class'].map(colors))

plt.show()