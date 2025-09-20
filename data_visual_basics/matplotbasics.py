import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Quick Recap:
# plt.plot() → line plot
# plt.scatter() → scatter plot
# plt.bar() → bar chart
# plt.hist() → histogram
# plt.subplot() → multiple plots
# plt.legend(), plt.xlabel(), plt.ylabel(), plt.title() → styling & labels


# Sample data
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Create a line plot
plt.plot(x, y)

# Show the plot
plt.show()

# Labels, Title, and Grid
plt.plot(x, y)
plt.xlabel("X-axis (Input)")
plt.ylabel("Y-axis (Output)")
plt.title("Simple Line Plot")
plt.grid(True)   # show grid lines
plt.show()

# Different Plot Types

# Line Plot
plt.plot(x, y, color="red", linestyle="--", marker="o")
plt.show()

# Scatter Plot
plt.scatter(x,y, color="red")
plt.show()

# Bar Chart
x = ["A", "B", "C", "D"]
y = [3, 7, 5, 9]
plt.bar(x, y, color="green")
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, color="purple", edgecolor="black")
plt.show()

# Multiple Plots in One Figure
x = [1,2,3,4]
y1 = [1,4,9,16]
y2 = [1,2,3,4]

plt.plot(x, y1, label="y=x^2")
plt.plot(x, y2, label="y=x")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Multiple Lines")
plt.legend()   # show labels
plt.show()

# Subplots (Grid of Plots)
x = [1,2,3,4]
y = [1,4,9,16]

plt.subplot(1,2,1)   # 1 row, 2 columns, position 1
plt.plot(x, y)

plt.subplot(1,2,2)   # 1 row, 2 columns, position 2
plt.bar(x, y)

plt.show()

# Saving Figures
plt.plot(x, y)
plt.savefig("my_plot.png")  # save as PNG

