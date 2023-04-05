#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
# Generate data on commute times.
size, scale = 1000, 10
commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)
commutes.plot.hist(grid=True, bins= 10, rwidth=0.9,
                  color='red')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)


# In[13]:


import pandas as pd
df = pd.read_csv("crashes.csv")
df2 = pd.read_csv("errors.csv")


print(df)

print(df2)

#df = pandas.DataFrame(list(zip(*Y)), columns=['Average_error', 'Crash'])
#df.apply(pandas.value_counts).plot.bar()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("crashes.csv")
df2 = pd.read_csv("errors.csv")

N = 3
ind = np.arange(N)
width = 0.25

xvals = pd.reset_option(df)
bar1 = plt.bar(ind, xvals, width, color = 'r')

yvals = pd.read_option(df2)
bar2 = plt.bar(ind+width, yvals, width, color='r')



plt.xlabel("Average time")
plt.ylabel('Crashes')
plt.title("Players Score")

plt.xticks(ind+width,['Sequence 1', 'Sequence 2', 'Sequence 3'])
plt.legend( (bar1, bar2), ('Crashes', 'Average time') )
plt.show()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt

N = 3
ind = np.arange(N)
width = 0.25

xvals = [8, 9, 2]
bar1 = plt.bar(ind, xvals, width, color = 'r')

yvals = [10, 20, 30]
bar2 = plt.bar(ind+width, yvals, width, color='g')



plt.xlabel("Average time")
plt.ylabel('Crashes')
plt.title("Players Score")

plt.xticks(ind+width,['Sequence 1', 'Sequence 2', 'Sequence 3'])
plt.legend( (bar1, bar2), ('Crashes', 'Average time') )
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("crashes.csv")
df2 = pd.read_csv("errors.csv")

N = 3
ind = np.arange(N)
width = 0.25

xvals = pd.reset_option(df)
bar1 = plt.bar(ind, xvals, width, color = 'r')

yvals = pd.read_option(df2)
bar2 = plt.bar(ind+width, yvals, width, color='r')



plt.xlabel("Average time")
plt.ylabel('Crashes')
plt.title("Players Score")

plt.xticks(ind+width,['Sequence 1', 'Sequence 2', 'Sequence 3'])
plt.legend( (bar1, bar2), ('Crashes', 'Average time') )
plt.show()

