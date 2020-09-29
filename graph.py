#!/usr/bin/env python
# coding: utf-8

# In[37]:


import RiskParity as rp
import numpy as np
import matplotlib.pyplot as plt
from WindPy import *
from datetime import datetime
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# In[67]:


MF_data = w.wsd("H11025.CSI", "close", "ED-2Y", datetime.today().strftime("%Y-%m-%d"), "Period=M")
MF_index = np.array(MF_data.Data).flatten()
MF_nv = np.divide(MF_index, MF_index[0])
MF_nv
MF_times = np.array([x.strftime("%Y-%m-%d") for x in MF_data.Times])
MF_times


# In[85]:


assets = "000905.SH,HSI.HI,SPX.GI,AU9999.SGE,H11008.CSI"
assets_data = w.wsd(assets, "close", "ED-2Y", datetime.today().strftime("%Y-%m-%d"),"Period=M")
assets_index = np.array(assets_data.Data)

def find_nv(x):
    return np.divide(x, x[0])

def nv_weighted_mean(x):
    return sum(np.multiply(x, rp.asset_allocation))

assets_nv = np.apply_along_axis(find_nv, axis=1, arr=assets_index )
assets_weighted_nv = np.apply_along_axis(nv_weighted_mean, axis = 0 , arr = assets_nv)



# In[57]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


plt.plot(MF_times, MF_nv, assets_weighted_nv)
plt.show()

# In[ ]:




