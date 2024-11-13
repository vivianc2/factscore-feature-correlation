from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import json

def build_probability_matrix_from_list(data, feature_set):
    lemmatizer = WordNetLemmatizer()
    feature_counts = Counter()
    co_occurrence_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each occupation group in the data
    for summary in tqdm(data, desc="Processing occupations"):
        words = word_tokenize(summary.lower())
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
        features_in_summary = [word for word in lemmatized_words if word in feature_set]

        in_this_summary = set()

        for i in range(len(features_in_summary)):
            for j in range(i + 1, len(features_in_summary)):
                if (features_in_summary[i],features_in_summary[j]) not in in_this_summary:
                    in_this_summary.add((features_in_summary[i], features_in_summary[j]))
                    feature_counts[features_in_summary[i]] += 1
                    co_occurrence_counts[features_in_summary[i]][features_in_summary[j]] += 1

    feature_list = sorted(feature_set)
    matrix_size = len(feature_list)
    probability_matrix = np.zeros((matrix_size, matrix_size))
    
    for i, f1 in enumerate(feature_list):
        total_co_occurrences = sum(co_occurrence_counts[f1].values())
        for j, f2 in enumerate(feature_list):
            # if f1 != f2 and total_co_occurrences > 0:
            if total_co_occurrences > 0:
                probability_matrix[i, j] = co_occurrence_counts[f1][f2] / total_co_occurrences

    return feature_list, probability_matrix, feature_counts, co_occurrence_counts


def save_outputs(feature_list, probability_matrix, feature_counts, co_occurrence_counts, output_prefix):
    # Save the feature list
    with open(f'{output_prefix}_feature_list.json', 'w') as f:
        json.dump(feature_list, f, indent=4)
    
    # Save the probability matrix using NumPy's save function
    np.save(f'{output_prefix}_probability_matrix.npy', probability_matrix)
    
    # Save the feature counts
    with open(f'{output_prefix}_feature_counts.json', 'w') as f:
        json.dump(dict(feature_counts), f, indent=4)
    
    # Save the co_occurrence counts
    with open(f'{output_prefix}_co_occurrence_counts.json', 'w') as f:
        # Convert defaultdict to a normal dictionary for JSON serialization
        co_occurrence_dict = {k: dict(v) for k, v in co_occurrence_counts.items()}
        json.dump(co_occurrence_dict, f, indent=4)


def save_sorted_probabilities(feature_list, probability_matrix, feature_counts, co_occurrence_counts, output_file):
    feature_index = {feature: idx for idx, feature in enumerate(feature_list)}
    sorted_probabilities = []

    # Collect all relevant data
    for f1 in feature_list:
        for f2 in feature_list:
            f1_idx = feature_index[f1]
            f2_idx = feature_index[f2]
            prob = probability_matrix[f1_idx][f2_idx]
            f1_count = feature_counts[f1]
            co_occurrence = co_occurrence_counts[f1][f2]
            sorted_probabilities.append(((f1, f2), prob, f1_count, co_occurrence))

    # Sort by probability, descending
    sorted_probabilities.sort(key=lambda x: x[1], reverse=True)

    # Save to file
    with open(output_file, 'w') as file:
        for entry in sorted_probabilities:
            line = f"{entry[0]}: Probability={entry[1]:.4f}, Count of {entry[0][0]}={entry[2]}, Count of {entry[0][0]} followed by {entry[0][1]}={entry[3]}\n"
            file.write(line)
