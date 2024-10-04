import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Read and prepare the data
df = pd.read_csv("blog-posts\gw7-preview\manutd-spurs-xg.csv")
df['Minute'] = df['Minute'].astype(int)
df = df.sort_values("Minute")

# Separate dataframes for each team
mun_df = df[df["Team"] == "Manchester Utd"]
tot_df = df[df["Team"] == "Tottenham"]

# Add a row with 0 xG at minute 0 for both teams
mun_df = pd.concat([pd.DataFrame({'Minute': [0], 'xG': [0], 'Team': ['Manchester Utd']}), mun_df])
tot_df = pd.concat([pd.DataFrame({'Minute': [0], 'xG': [0], 'Team': ['Tottenham']}), tot_df])

# Calculate cumulative xG
mun_df['Cumulative_xG'] = mun_df['xG'].cumsum()
tot_df['Cumulative_xG'] = tot_df['xG'].cumsum()

# After creating the figure and before plotting
fig, ax = plt.subplots(figsize=(12,8))
plt.style.use('dark_background')

# Set the figure facecolor to black
fig.patch.set_facecolor('black')

# Set the axes facecolor to black
ax.set_facecolor('black')

# Plot your data as before, but use ax instead of plt
ax.step(mun_df["Minute"], mun_df["Cumulative_xG"], where="post", label="Man Utd xG", color='#DA291C')
ax.step(tot_df["Minute"], tot_df["Cumulative_xG"], where="post", label="Tottenham xG", color='#00509d')

# Customize the plot using ax
ax.set_xlabel("Minute", color="white")
x_ticks = range(0, max(df['Minute'])+1, 15)
ax.set_xticks(x_ticks)
ax.tick_params(axis='x', colors='white')
ax.set_ylabel("xG", color="white")
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Adjust the top margin of the plot
plt.subplots_adjust(top=0.85)

# Set x-axis limits
ax.set_xlim(-5, max(df['Minute'])+5)

# Set y-axis limits and ticks
y_max = max(mun_df["Cumulative_xG"].max(), tot_df["Cumulative_xG"].max())
ax.set_ylim(-0.5, y_max + 0.5)

# Create y-ticks
y_ticks = np.arange(0, y_max + 0.5, 0.5)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])

# Make y-tick labels white
ax.tick_params(axis='y', colors='white')

# Show grid
ax.grid(True, linestyle='--', alpha=0.3, color='white')

# Add final xG values as text annotations
ax.text(mun_df["Minute"].iloc[-1], mun_df["Cumulative_xG"].iloc[-1], f'{mun_df["Cumulative_xG"].iloc[-1]:.2f}',
        verticalalignment='bottom', horizontalalignment='right', color='#DA291C')
ax.text(tot_df["Minute"].iloc[-1], tot_df["Cumulative_xG"].iloc[-1], f'{tot_df["Cumulative_xG"].iloc[-1]:.2f}',
        verticalalignment='bottom', horizontalalignment='right', color='#00509d')

plt.show()