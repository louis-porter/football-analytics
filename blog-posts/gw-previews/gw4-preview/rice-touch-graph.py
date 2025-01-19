import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

logo= r"C:\Users\Owner\dev\football-analytics\plots\logos\arsenal.png"

# Load the CSV data
df = pd.read_csv(r"C:\Users\Owner\dev\football-analytics\blog-posts\gw4-preview\dec_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Tou/90 5 rolling"])

# Fitting a linear trend
x = df.index
y = df["Tou/90 5 rolling"]
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Calculate the percentage decrease in trend
start_value = p(x[0])
end_value = p(x[-1])
percentage_decrease = (end_value - start_value) / start_value * 100

# Apply the dark background style
plt.style.use("dark_background")

# Creating the line graph
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

# Plotting the data and trend line
ax.plot(df.index, df["Tou/90 5 rolling"], label="Touches per 90 Mins", color="#EF0107") 
ax.plot(x, p(x), color="#FE484E", linestyle="--", label="Trend Line", alpha=.8) 
ax.axvline(x=46, color="white", linestyle="--", alpha=.65, label="24-25 Season", dashes=(6,6)) 

# Adding labels to plot
ax.text(47, 95, "24-25 Season", rotation=90, verticalalignment='center', color='white', 
         fontsize=7, fontdict={"fontweight": "bold"}, alpha=0.7)
trend_label_x_position = x[-1]
trend_label_y_position = p(x[-1]) -.5
ax.text(trend_label_x_position, trend_label_y_position, f"{percentage_decrease:.2f}%", 
         horizontalalignment='left', color='#FE484E', fontsize=7, alpha=.8, weight="bold")

# Customizing appearance
sns.despine(left=True, bottom=True)
ax.grid(True, axis='y', color='gray')

# Adjust labels and ticks
ax.set_xlabel("")
ax.set_xticks([])

ax.set_ylabel("Touches per 90 Mins", color="white")

# Titles
ax.set_title("Declan Rice's touches per game have decreased since joining Arsenal", x=0.39, pad=40, fontdict={"fontsize": 14, "fontweight": "bold"})
fig.suptitle("Touches per 90 mins rolling 5-game average | Premier League and Champions League", x=0.368, y=.915, size=10, weight="bold", alpha=0.76 )

prem_logo = OffsetImage(plt.imread(logo), zoom=0.25)
ab = AnnotationBbox(prem_logo, xy=(0.95, 1.09), xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5))
ax.add_artist(ab)

plt.show()

print(df.shape)