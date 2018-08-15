import pandas as pd 
import numpy as np
from numpy.linalg import inv
def manhalo(x,y,t):
    vai=t.cov()
    ivai=inv(vai)
    z=x-y
    z=np.matrix(z)
    arr=np.matrix(z)
    arr_transpose= arr.T
    d1= z * ivai
    ans= d1 * arr_transpose
    return ans
df=pd.read_csv("iris.csv", names=["seplen","sepwid","petlen","petwid","class"])
a=np.array([4.6,3.1,1.5,0.2])
b=np.array([5.0,3.5,1.46,0.2540])
d= df[['seplen','sepwid','petlen','petwid']]
print ("the manhalova distance= \n",manhalo(a,b,d))
