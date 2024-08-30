import matplotlib.pyplot as plt
import numpy as np

# Attributes
labels = ['Passing', 'Shooting', 'Dribbling', 'Defending', 'Physicality', 'Pace']

# Example player data
player_data = [75, 85, 90, 60, 80, 85]

# Number of variables
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is made in a circular (polar) space, so we need to "close the loop"
player_data += player_data[:1]
angles += angles[:1]

# Set the font for the plot
plt.rcParams.update({
    'font.family': 'arial'
})

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw the outline of the radar chart
ax.plot(angles, player_data, linewidth=2, linestyle='solid')
ax.fill(angles, player_data, 'b', alpha=0.25)

# Add labels for each point with correct rotation
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='black')

# Rotate labels to face outward
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    x, y = label.get_position()
    if 0.25 < angle <= np.pi / 2:
        label.set_horizontalalignment('left')
        label.set_verticalalignment('bottom')
    elif np.pi / 2 < angle <= np.pi:
        label.set_horizontalalignment('left')
        label.set_verticalalignment('top')
    elif np.pi < angle <= 3 * np.pi / 2:
        label.set_horizontalalignment('right')
        label.set_verticalalignment('top')
    else:
        label.set_horizontalalignment('right')
        label.set_verticalalignment('bottom')

    # Increase distance of labels from the plot by adjusting their position
    label.set_position((x, y + 0.1))

# Show the radar chart
plt.show()
