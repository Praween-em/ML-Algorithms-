# Define the main ID3 algorithm
def id3(data, attributes, target_attribute):
    # If all examples have the same target value, return that value
    unique_targets = set(row[target_attribute] for row in data)
    if len(unique_targets) == 1:
        return next(iter(unique_targets))

    # If the attribute list is empty, return the most common target value
    if not attributes:
        return max(set(row[target_attribute] for row in data), key=list(row[target_attribute] for row in data).count)

    # Otherwise, find the best attribute to split on
    best_attribute = find_best_attribute(data, attributes,target_attribute)
    decision_tree = {best_attribute: {}}
    split_data_dict = split_data(data, attributes.index(best_attribute))
    new_attributes = [attr for attr in attributes if attr != best_attribute]

    for value, subset in split_data_dict.items():
        decision_tree[best_attribute][value] = id3(subset, new_attributes, target_attribute)

    return decision_tree

# Sample data (replace with your own dataset)
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Strong', 'No']
]

attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target_attribute = -1

# Build the decision tree
decision_tree = id3(data, attributes, target_attribute)

# Define a function to classify a new sample using the decision tree
def classify(sample, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0]
    value = sample[attributes.index(attribute)]
    subtree = tree[attribute][value]
    return classify(sample, subtree)

# Test a new sample
new_sample = ['Sunny', 'Hot', 'High', 'Strong']
classification = classify(new_sample, decision_tree)
print("Classification for the new sample:", classification)
