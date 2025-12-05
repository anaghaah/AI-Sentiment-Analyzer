import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. SAMPLE DATA (A list of 10 fake customer reviews)
reviews = [
    "The product is absolutely amazing and exceeded all my expectations.", # Positive
    "It arrived broken, and the customer service was useless. Terrible experience.", # Negative
    "This laptop bag is black, and it fits a 15-inch computer.", # Neutral
    "The packaging was neat and delivery was very fast. Great purchase!", # Positive
    "I'm deeply disappointed with the quality and the high price.", # Negative
    "It works as described, no problems and no surprises.", # Neutral/Slightly Positive
    "Totally worth the money, five stars!", # Positive
    "It's okay, nothing special, but it gets the job done.", # Neutral
    "I wouldn't recommend this to anyone. A complete failure.", # Negative
    "The design is elegant, though the color is slightly off." # Mixed/Neutral
]

# 2. Setup the Analyzer
sia = SentimentIntensityAnalyzer()
results = [] # List to store the results

# 3. Processing Loop
print("--- Starting Sentiment Analysis ---")
for review in reviews:
    # Get the scores
    score = sia.polarity_scores(review)
    
    # Determine the Final Sentiment Label based on the Compound Score
    if score['compound'] >= 0.05:
        label = 'Positive'
    elif score['compound'] <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    
    # Store the results
    results.append({
        'Review': review,
        'Compound_Score': score['compound'],
        'Sentiment_Label': label
    })
    print(f"Review: '{review[:40]}...' -> Label: {label}") # Print a short summary

# 4. Convert Results to a Data Table (DataFrame)
df = pd.DataFrame(results)
print("\n--- Final Data Table (DataFrame) ---")
print(df)


# 5. Data Visualization (The powerful step for your project!)
sentiment_counts = df['Sentiment_Label'].value_counts()

# Create a bar chart
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribution of Customer Sentiments')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0) # Keeps labels horizontal
plt.grid(axis='y', linestyle='--')
plt.show() # This displays the final chart!

print("--- Analysis Complete: A bar chart has been generated! ---")