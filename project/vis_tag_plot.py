'''
Author: Alex Shi
Date: 2021-12-05 11:43:01
LastEditTime: 2021-12-05 15:31:51
LastEditors: Alex Shi
Description: 
FilePath: /Course Paper/vis_tag_plot.py
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.polynomial import poly
import pandas as pd
from math import pi
from numpy.lib.function_base import angle, place


df = pd.read_csv('./data/tags_count.csv')
df = df.sort_values(by=['Unnamed: 0'], na_position='first')
df.columns = ['Tags', 'Dune', 'Space Odyssey']
df.columns.name = 'Index'
df.drop(index=[35, 32], inplace=True)
df[['Dune', 'Space Odyssey']] = df[['Dune', 'Space Odyssey']] / df[['Dune', 'Space Odyssey']].sum()
df.set_index('Tags', inplace=True) 
categories = list(df.transpose())[1:]
N = len(categories)

angles = [n/float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories)
ran = np.arange(0.0, 0.2, 0.05)
ax.set_rlabel_position(0)
plt.yticks(ran, [str(np.round(item, 2)) for item in ran], color='grey', size=7)
plt.ylim(0, 0.2)


values1 = df['Dune']
ax.plot(angles, values1, linewidth=1, linestyle='solid', label='Dune')
ax.fill(angles, values1, 'b', alpha=0.1)

values2 = df['Space Odyssey']
ax.plot(angles, values2, linewidth=1, linestyle='solid', label='Space Odyssey')
ax.fill(angles, values2, 'r', alpha=0.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Ratio of different kinds of words')
plt.show()