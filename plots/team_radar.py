from mplsoccer import Radar, grid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logo_path = r"plots\logos\man-utd.png"

# Load the logo image
logo_img = mpimg.imread(logo_path)

# creating the figure using the grid function from mplsoccer:
fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# Setup the radar object
radar = Radar(params=['xG', 'Shots For', 'Touches in Opp Penalty Area', "Possession",
                      "xGC", "Shots Against", "PPDA"], 
              lower_is_better=["xGC", "Shots Against", "PPDA"], 
              min_range=[0.6, 7, 13.2, 37.4, 0.72, 6.8, 8.8], 
              max_range=[2.27, 19.6, 46.4, 65, 2.18, 18.8, 17.7])

# plot the radar
radar.setup_axis(ax=axs['radar'], facecolor='None')
rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#28252c', edgecolor='#39353f', lw=1.5)

# First radar values
values_1 = [1.9, 13.6, 27.8, 56.6, 1.28, 9.8, 10.9]

# Draw the first radar
radar_output_1 = radar.draw_radar(values_1, ax=axs['radar'],
                                   kwargs_radar={'facecolor': '#2D22C9', 'alpha': 0.6},
                                   kwargs_rings={'facecolor': '#2D22C9', 'alpha': 0.6})

# Second radar values (for overlay)
values_2 = [1.49, 14.3, 27, 50.6, 1.81, 17.4, 12.6]

# Draw the second radar (overlay)
radar_output_2 = radar.draw_radar(values_2, ax=axs['radar'],
                                   kwargs_radar={'facecolor': '#DF1111', 'alpha': 0.6},  # Different color
                                   kwargs_rings={'facecolor': '#DF1111', 'alpha': 0.6})

# Add range and parameter labels
range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=12, color='#fcfcfc')
param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=12, color='#fcfcfc')

# Add the title and subtitle
title1_text = axs['title'].text(0.01, 0.65, 'Manchester United Have Improved So Far this Season', fontsize=20,
                                ha='left', va='center', color='#e4dded', fontdict={"fontweight":"bold"})


# Breaking down the title2_text for different colors
axs['title'].text(0.011, 0.25, 'Premier League | ', fontsize=15, ha='left', va='center', color='#fcfcfc')
axs['title'].text(0.011 + 0.145, 0.25, '2023/24', fontsize=15, ha='left', va='center', color='#DF1111')  # Blue
axs['title'].text(0.011 + 0.215, 0.25, ' & ', fontsize=15, ha='left', va='center', color='#fcfcfc')  # White
axs['title'].text(0.011 + 0.24, 0.25, '2024/25', fontsize=15, ha='left', va='center', color='#2D22C9')  # Red

# Set the background color for the figure
fig.set_facecolor('#121212')

# Show the plot
plt.show()
