
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re

stereotype_jokes = [
    "Women are terrible drivers; they always forget their turn signals!",
    "Why do blondes always have dumb moments? Too much hair, not enough brain!",
    "All politicians are liars—it's like a requirement for the job.",
    "Engineers have no social skills; they’re only comfortable around machines.",
    "Why don’t men ever ask for directions? They’d rather be lost than admit they’re wrong.",
    "All Italians talk with their hands—take that away, and they couldn’t speak!",
    "Why don’t lawyers go to heaven? Because they're already familiar with hell!",
    "Why are all Irish people heavy drinkers? Must be something in the Guinness.",
    "Why are mathematicians bad at relationships? Because they’re always calculating instead of feeling.",
    "Why are French people so rude? They think their language is the only one that matters.",
    "How many blondes does it take to change a light bulb? None—they'd rather call someone to do it.",
    "Why do all Germans love rules? It's like they’re born with a manual in their hands!",
    "All teenagers are lazy—they think chores are beneath them.",
    "Why don’t programmers ever shower? Because they don't want to wash away their bugs!",
    "All mothers are overprotective—it’s like they think you’ll break the moment they look away.",
    "Why are all British people obsessed with tea? It’s like they think it solves everything.",
    "Asians are great at math—just hand them a calculator and step back.",
    "All gamers are basement-dwelling loners with no life outside the screen.",
    "Millennials are lazy—they expect everything handed to them without working for it.",
    "All dads are the same—completely clueless when it comes to technology.",
]

non_stereotype_jokes = [
    "I used to be a baker, but I couldn’t make enough dough.",
    "Why don’t skeletons fight each other? They don’t have the guts!",
    "I’m reading a book on anti-gravity—it’s impossible to put down.",
    "I once told a chemistry joke, but there was no reaction.",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "Parallel lines have so much in common. It’s a shame they’ll never meet.",
    "I went to a seafood disco last night and pulled a mussel.",
    "Why did the bicycle fall over? It was two-tired.",
    "I tried to catch some fog, but I mist.",
    "What do you call fake spaghetti? An impasta.",
    "I told my computer I needed a break, and now it’s not talking to me.",
    "I’m on a whiskey diet. I’ve lost three days already.",
    "I wanted to be a doctor, but I didn’t have the patients.",
    "I was going to make a belt out of watches, but it was a waist of time.",
    "I used to be afraid of hurdles, but I got over it.",
    "I was wondering why the ball kept getting bigger, and then it hit me.",
    "The kleptomaniac couldn’t help himself—he just took everything for granite.",
    "I told my friend 10 jokes to make him laugh. Sadly, no pun in ten did.",
    "Why did the coffee file a police report? It got mugged.",
    "Time flies like an arrow. Fruit flies like a banana.",
    "Why do all programmers prefer dark mode? Because light attracts bugs!"
]

# Create a combined dataset
data = {
    "text": stereotype_jokes + non_stereotype_jokes,
    "category": ["Stereotype"] * len(stereotype_jokes) + ["Non-Stereotype"] * len(non_stereotype_jokes)
}

# Converting the data to a DataFrame
df = pd.DataFrame(data)

# Vectorizing the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Defining the target variable
y = df['category']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using Multinomial Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nConfusion Matrix:\n", conf_matrix)

# Plotting the Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Function to preprocess jokes
def preprocess_joke(joke):
    joke = re.sub(r'[^\w\s]', '', joke)  # Remove punctuation and special characters
    joke = joke.lower().strip()  # Lowercase and remove extra whitespace
    if not joke:
        raise ValueError("The input joke is empty after preprocessing. Please provide valid text.")
    return joke

# Function to predict category for a new joke
def predict_category(joke):
    # Preprocess the input joke
    joke_cleaned = preprocess_joke(joke)
    
    # Transform the cleaned joke into the vectorized format
    joke_vectorized = vectorizer.transform([joke_cleaned])
    
    # Predict the category of the joke
    prediction = model.predict(joke_vectorized)
    
    return prediction[0]

# Example usage
joke = "Instead of being a Robin Hood, I ended up robbing the hood."
try:
    print(f"The predicted category for the joke is: {predict_category(joke)}")
except ValueError as e:
    print(e)

