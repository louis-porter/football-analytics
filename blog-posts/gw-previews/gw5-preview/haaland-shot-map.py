import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

haaland_img_path = "blog-posts\gw5-preview\haaland.png"
mancity_img_path = "plots\logos\man-city.png"


shots = pd.read_csv("blog-posts\gw5-preview\HaalandShots.csv")
shots = shots[shots["season"] == 2024]

print(shots)

# Graph Creation

shots["x"] = shots["x"] * 100
shots["y"] = shots["y"] * 100

total_shots = shots.shape[0]
total_goals = shots[shots["result"] == "Goal"].shape[0]
total_xG = shots["xG"].sum()
xG_per_shot = total_xG / total_shots
points_average_distance = shots['x'].mean()
actual_average_distance = 120 - (shots['x'] * 1.2).mean()
print(points_average_distance, actual_average_distance)




# Color -> They went with a black so we'll do the same. Notice how it isn't a pure black but rather a lighter little bit of grey black
background_color='#0C0D0E'

# Font -> Fonts are tricky since they have their "brand fonts" so we'll just use something similar
import matplotlib.font_manager as font_manager




pitch = VerticalPitch(
    pitch_type='opta', 
    half=True, 
    pitch_color=background_color, 
    pad_bottom=.5, 
    line_color='white',
    linewidth=.75,
    axis=True, label=True
)

# create a subplot with 2 rows and 1 column
fig = plt.figure(figsize=(8, 12))
fig.patch.set_facecolor(background_color)


# Top row for the team names and score
# [left, bottom, width, height]

ax1 = fig.add_axes([0, 0.7, 1, .2])
ax1.set_facecolor(background_color)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)


ax1.text(
    x=0.06, 
    y=.55, 
    s='Erling Haaland has exceeded his xG by 4.46 goals', 
    fontsize=15, 
    fontweight='bold', 
    color='white', 
    ha='left'
)
ax1.text(
    x=0.06, 
    y=.4, 
    s=f'Shots | Premier League 2024-25', 
    fontsize=12,
    fontweight='bold',
    color='white', 
    ha='left',
    alpha=0.8
)

ax1.text(
    x=0.45, 
    y=0.2, 
    s=f'Goal', 
    fontsize=10,
    color='white', 
    ha='right'
)
ax1.scatter(
    x=0.47, 
    y=0.23, 
    s=100, 
    color='#6CABDD', 
    edgecolor='white', 
    linewidth=.8,
    alpha=.7
)


ax1.scatter(
    x=0.53, 
    y=0.23, 
    s=100, 
    color=background_color, 
    edgecolor='white', 
    linewidth=.8
)

ax1.text(
    x=0.55, 
    y=0.2, 
    s=f'No Goal', 
    fontsize=10,
    color='white', 
    ha='left'
)

ax1.set_axis_off()


ax2 = fig.add_axes([.05, 0.25, .9, .5])
ax2.set_facecolor(background_color)

pitch.draw(ax=ax2)


# create a scatter plot at y 100 - average_distance
ax2.scatter(
    x=90, 
    y=points_average_distance, 
    s=100, 
    color='white',  
    linewidth=.8
)
# create a line from the bottom of the pitch to the scatter point
ax2.plot(
    [90, 90], 
    [100, points_average_distance], 
    color='white', 
    linewidth=2
)

# Add a text label for the average distance
ax2.text(
    x=90, 
    y=points_average_distance - 4, 
    s=f'Average Distance\n{actual_average_distance:.1f} yards', 
    fontsize=10, 
    color='white', 
    ha='center'
)


for x in shots.to_dict(orient='records'):
    pitch.scatter(
        (x["x"]), 
        (x["y"]), 
        s=300 * x['xG'], 
        color='#6CABDD' if x['result'] == 'Goal' else background_color, 
        ax=ax2,
        alpha=.7,
        linewidth=.8,
        edgecolor='white'
    )
    
ax2.set_axis_off()

# add another axis for the stats
ax3 = fig.add_axes([0, .2, 1, .05])
ax3.set_facecolor(background_color)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

ax3.text(
    x=0.25, 
    y=.5, 
    s='Shots', 
    fontsize=15,
    fontweight='bold', 
    color='white', 
    ha='left'
)

ax3.text(
    x=0.25, 
    y=0, 
    s=f'{total_shots}', 
    fontsize=13,
    color='#6CABDD', 
    ha='left'
)

ax3.text(
    x=0.38, 
    y=.5, 
    s='Goals', 
    fontsize=15,
    fontweight='bold', 
    color='white', 
    ha='left'
)

ax3.text(
    x=0.38, 
    y=0, 
    s=f'{total_goals}', 
    fontsize=13,
    color='#6CABDD', 
    ha='left'
)

ax3.text(
    x=0.53, 
    y=.5, 
    s='xG', 
    fontsize=15,
    fontweight='bold', 
    color='white', 
    ha='left'
)

ax3.text(
    x=0.53, 
    y=0, 
    s=f'{total_xG:.2f}', 
    fontsize=13,
    color='#6CABDD', 
    ha='left'
)

ax3.text(
    x=0.63, 
    y=.5, 
    s='xG/Shot', 
    fontsize=15,
    fontweight='bold', 
    color='white', 
    ha='left'
)

ax3.text(
    x=0.63, 
    y=0, 
    s=f'{xG_per_shot:.2f}', 
    fontsize=13,
    color='#6CABDD', 
    ha='left'
)

ax3.set_axis_off()

# Add Haaland Image

haaland_img = mpimg.imread(haaland_img_path)
imagebox = OffsetImage(haaland_img, zoom=0.1) 
image_ab = AnnotationBbox(imagebox, (20, 63.5), frameon=False)
ax2.add_artist(image_ab)

mancity_img = mpimg.imread(mancity_img_path)
imagebox = OffsetImage(mancity_img, zoom=0.3) 
image_ab = AnnotationBbox(imagebox, (0.9, 0.49), frameon=False)
ax1.add_artist(image_ab)



plt.show()