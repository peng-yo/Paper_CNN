import seaborn as sns
import pandas as pd

# Create a DataFrame to store the data
data = pd.DataFrame(
    {
        "Kernel Size": ["3", "3", "3", "3", "5", "5", "5", "5"],
        "Filters": [16, 32, 64, 128, 16, 32, 64, 128],
        "Recall Rate": [
            0.9911968723296575,
            0.9914481566409913,
            0.9920216828367148,
            0.9916713082738691,
            0.9933660960602501,
            0.9918008116794962,
            0.9941875715657572,
            0.992092887239927,
        ],
    }
)

# Set the style of the plot
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Create the line plot
g = sns.relplot(
    data=data,
    x="Filters",
    y="Recall Rate",
    hue="Kernel Size",
    kind="line",
    style="Kernel Size",
    markers=True,
    dashes=False,
)

# Set the title and axes labels
g.fig.suptitle("Recall Rates by Kernel Size and Number of Filters", fontsize=16)
g.set_axis_labels("Number of Filters", "Recall Rate")

# Add legend title and adjust legend position
g._legend.set_title("Kernel Size")
g._legend.set_bbox_to_anchor([1, 0.75])

# Save the plot as a PNG file
g.savefig("recall_rates.png", dpi=300)
