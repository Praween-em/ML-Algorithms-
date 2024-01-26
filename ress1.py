import pandas as pd

def initialize_hypotheses(n):
    # Initialize the hypothesis space with the most specific and most general hypotheses
    specific_hypothesis = ["0"] * n
    general_hypothesis = ["?"] * n
    return specific_hypothesis, general_hypothesis

def is_consistent(instance, hypothesis):
    # Check if the instance is consistent with the hypothesis
    for i in range(len(instance) - 1):  # Exclude the target concept in the last column
        if hypothesis[i] != "?" and hypothesis[i] != instance[i]:
            return False
    return True

def candidate_elimination_algorithm(data):
    # Initialize hypotheses
    n_attributes = len(data.columns) - 1
    specific_hypothesis, general_hypothesis = initialize_hypotheses(n_attributes)

    # Iterate through the training examples
    for index, row in data.iterrows():
        if row.iloc[-1] == "yes":  # Positive example
            for i in range(n_attributes):
                if specific_hypothesis[i] == "0":
                    specific_hypothesis[i] = row[i]
                elif specific_hypothesis[i] != row[i]:
                    specific_hypothesis[i] = "?"

            # Remove inconsistent general hypotheses
            for i in range(n_attributes):
                if specific_hypothesis[i] != general_hypothesis[i]:
                    general_hypothesis[i] = "?"

        else:  # Negative example
            for i in range(n_attributes):
                if row[i] != specific_hypothesis[i]:
                    general_hypothesis[i] = specific_hypothesis[i]

    return specific_hypothesis, general_hypothesis

# Assuming the CSV file has headers and the target concept is in the last column
file_path = "enjoysport.csv"
data = pd.read_csv(file_path)

# Execute the Candidate-Elimination algorithm
specific_hypothesis, general_hypothesis = candidate_elimination_algorithm(data)

# Display the final hypotheses
print("Specific Hypothesis:", specific_hypothesis)
print("General Hypothesis:", general_hypothesis)
