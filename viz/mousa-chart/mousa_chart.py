import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def create_venn_diagram(scores, player_name="Player"):
    """
    Create a Venn diagram with three circles whose sizes are determined by input scores.
    
    Args:
        scores (list): Three values between 1-100 determining circle sizes
        player_name (str): Name of the player to display in title
        
    Returns:
        None (displays the plot)
    """
    if not all(1 <= score <= 100 for score in scores):
        raise ValueError("All scores must be between 1 and 100")
    
    # Create figure and axis with dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#1a1a1a')  # Dark grey background
    ax.set_facecolor('#1a1a1a')
    
    # Calculate radii based on scores
    radii = [score/500 for score in scores]
    
    # Centers positioned in a tighter equilateral triangle for proper overlap
    centers = [
        (0.5 - 0.12, 0.5 - 0.08),             # Bottom left
        (0.5 + 0.12, 0.5 - 0.08),             # Bottom right
        (0.5, 0.5 + 0.12)                      # Top
    ]
    
    # Colors for circles and labels (brighter colors for dark mode)
    colors = ['#ff333366', '#33ff3366', '#3333ff66']  # Semi-transparent RGB for circles
    label_colors = ['#ff5555', '#55ff55', '#5555ff']  # Brighter RGB for labels
    circle_labels = ['Ball Winning', 'Ball Progession', 'Ball Retention']
    
    # Create and add circles individually for proper overlap
    for center, radius, color in zip(centers, radii, colors):
        circle = Circle(center, radius, facecolor=color, edgecolor='white', alpha=0.5)
        ax.add_patch(circle)
    
    # Calculate label positions based on circle sizes
    margin = 0.05  # Additional margin from circle edge
    label_positions = [
        (0.5 - 0.12, centers[0][1] - radii[0] - margin),    # Below left circle
        (0.5 + 0.12, centers[1][1] - radii[1] - margin),    # Below right circle
        (0.5, centers[2][1] + radii[2] + margin)            # Above top circle
    ]
    
    for pos, label, color in zip(label_positions, circle_labels, label_colors):
        ax.text(pos[0], pos[1], label,
                color=color,
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='bold',
                fontsize=12)
    
    # Set plot limits and aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add scores as labels in white
    for i, (center, score) in enumerate(zip(centers, scores)):
        ax.text(center[0], center[1], f'{score}', 
                color='white',
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='bold')
    
    plt.title(f'{player_name}', color='white', pad=0, fontdict={"fontsize":22, "fontweight":"bold"})
    plt.show()

# Example usage
scores = [49, 95, 92]
create_venn_diagram(scores, "Martin Odegaard")