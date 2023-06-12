import pandas as pd
import matplotlib as plt

# create a sample DataFrame
ddf = pd.DataFrame([['www.example1.com',False],['www.google1.com',True],[ 'www.facebook.com',False],[ 'www.example.com',True],[ 'www.google.com',False],[ 'www.facebooc.com',True]],
                   columns=['qname','is_malicious'])

# define the colors for different values of is_malicious
color_map = {True: 'red', False: 'green'}
colors = ddf['is_malicious'].map(color_map)

# create the scatter plot
ax = ddf.plot(kind='scatter', x='qname', y=ddf.index, c=colors)

# set the title and labels
ax.set_title('Scatter plot of qname')
ax.set_xlabel('qname')
ax.set_ylabel('Index')

# show the plot
plt.show()