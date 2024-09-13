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

att_y = df["Att /90 roll"]
att_z = np.polyfit(x, att_y, 1)
att_p = np.poly1d(att_z)

def_y = df["Def 90 roll"]
def_z = np.polyfit(x, def_y, 1)
def_p = np.poly1d(def_z)

# Calculate the percentage decrease in trend
start_value = def_p(x[0])
end_value = def_p(x[-1])
percentage_decrease = (end_value - start_value) / start_value * 100

# Apply the dark background style
plt.style.use("dark_background")

# Creating the line graph
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

# Plotting the data and trend line
ax.plot(df.index, df["Def 90 roll"], label="Def 1/3 Touches per 90 Mins", color="#0C6FE9") 
ax.plot(x, def_p(x), color="#0C6FE9", linestyle="--", label="_nolegend_", alpha=.8) 

ax.plot(df.index, df["Att /90 roll"], label="Att 1/3 Touches per 90 Mins", color="#EF0107") 
ax.plot(x, att_p(x), color="#EF0107", linestyle="--", label="_nolegend_", alpha=.8) 

ax.axvline(x=46, color="white", linestyle="--", alpha=.65, label="_nolegend_" , dashes=(6,6)) 

# Adding labels to plot
ax.text(47, 27, "24-25 Season", rotation=90, verticalalignment='center', color='white', 
         fontsize=7, fontdict={"fontweight": "bold"}, alpha=0.7, label="_nolegend_")
trend_label_x_position = x[-1]
trend_label_y_position = def_p(x[-1]) -.5

# Customizing appearance
sns.despine(left=True, bottom=True)
ax.grid(True, axis='y', color='gray')

# Adjust labels and ticks
ax.set_xlabel("")
ax.set_xticks([])

ax.set_ylabel("Touches per 90 Mins", color="white")

# Titles
ax.set_title("Declan Rice has been less involved in build-up over time", x=0.30, pad=40, fontdict={"fontsize": 14, "fontweight": "bold"})
subtitle = ("<span style='color:#EF0107'>Att 1/3 Touches per 90</span> & "
            "<span style='color:#0C6FE9'>Def 1/3 Touches per 90</span> | "
            "Premier League & Champions League")
fig.text(0.03, 0.90, "Att 1/3 Touches per 90", color="#EF0107", fontsize=10, weight="bold", alpha=0.76)
fig.text(0.215, 0.90, "&", color="white", fontsize=10, weight="bold", alpha=0.76)
fig.text(0.237, 0.90, "Def 1/3 Touches per 90", color="#0C6FE9", fontsize=10, weight="bold", alpha=0.76)
fig.text(0.42, 0.90, "| Premier League & Champions League", color="white", fontsize=10, weight="bold", alpha=0.76)

prem_logo = OffsetImage(plt.imread(logo), zoom=0.25)
ab = AnnotationBbox(prem_logo, xy=(0.95, 1.09), xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5))
ax.add_artist(ab)

#ax.legend()

plt.show()
