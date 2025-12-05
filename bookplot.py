import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = np.loadtxt('book_project.csv', delimiter=",", dtype=str)

vocab_col = data[1:, 2]         
score_col = data[1:, 1]         
label_col = data[1:, 3]         

vocab_size = [float(i) for i in vocab_col]
reading_score = [float(i) for i in score_col]

sns.scatterplot(x=vocab_size, y=reading_score, hue=label_col)

plt.xlabel("Vocabulary Size (words)")
plt.ylabel("Reading Score")
plt.title("Reading Score vs Vocabulary Size")

slope, intercept, r_value, p_value, std_err = stats.linregress(vocab_size, reading_score)

x_line = np.linspace(min(vocab_size), max(vocab_size), 100)
y_line = slope * x_line + intercept

plt.plot(x_line, y_line)

plt.show()

