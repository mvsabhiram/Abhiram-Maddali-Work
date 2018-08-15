import numpy as np 
from math import*
from decimal import Decimal
import matplotlib.pyplot as plt
def n_pow(f,r_1):
    inv_r= 1/np.float(r_1);
    return round(Decimal(f) ** Decimal(inv_r),3)
    
def minkaw(x,y,r):
    if x.size == y.size:
        s=sum(pow(abs(a-b),r) for a,b in zip(x, y))
        ans=n_pow(s,r)
        return(ans);
    else:
        print "invalid data";

a=np.array([4.6,3.1,1.5,0.2])
b=np.array([5.0,3.5,1.46,0.2540])
x_1=minkaw(a,b,1);
x_2=minkaw(a,b,2);
x_100=minkaw(a,b,100);
print "for r=1 Minkowski distance=" ,x_1 
print "for r=2 Minkowski distance=",x_2
print "for r=100 Minkowski distance=",x_100
plt.scatter([1,2,100],[x_1,x_2,x_100])
plt.show()