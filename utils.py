from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import json

import os
import json
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import os
import json
from collections import defaultdict
import numpy as np

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


def preprocess_text(text):
    """Tokenize, lemmatize, and clean text by retaining only alphabetic tokens."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    
    # Define stopwords (including custom ones)
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'also', 'known', 'later', 'born', 'became', 'lxi', 'lx', '``', '-lrb-', '-rrb-', "'s", '--'}
    stop_words.update(custom_stop_words)
    
    cleaned = []
    for token in tokens:
        # Retain only tokens that are purely alphabetic and not in the stop words
        if token.isalpha() and token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            cleaned.append(lemma)
    return cleaned


def get_feature_probabilities(occupation_groups, top_n=50):
    """Calculate P(f|occupation) for top features, and construct co-occurrence matrices.
    
    Input:
      occupation_groups: dict mapping occupation to a dictionary of {entity_name: summary}
      top_n: number of top frequent features to consider per occupation
      
    Returns:
      feature_probs: dictionary mapping each occupation to a dict {feature: fraction of summaries with that feature}
      all_matrices: dictionary mapping each occupation to {'features': top_features, 'matrix': probability_matrix}
    """
    feature_probs = defaultdict(dict)
    pair_count = defaultdict(dict)
    occ_count = defaultdict(int)
    all_matrices = {}  # using a plain dict since we'll overwrite each occupation's matrix

    # Iterate over each occupation group
    for occupation, summaries in tqdm(occupation_groups.items(), desc="Processing occupations"):
        # Collect all words from all summaries for this occupation
        all_words = []
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))
        feature_counts = defaultdict(int)
        
        for summary in summaries.values():
            all_words.extend(preprocess_text(summary))
            
        word_counts = Counter(all_words)
        top_features = [word for word, _ in word_counts.most_common(top_n)]
        
        # Initialize pair counts and co-occurrence counts for each top feature
        for word in top_features:
            pair_count[occupation][word] = 0
            for word2 in top_features:
                co_occurrence_counts[word][word2] = 0
        
        # Process each summary for this occupation
        for summary in summaries.values():
            s = preprocess_text(summary)
            occ_count[occupation] += 1
            # Increase count if a top feature appears in this summary (unique occurrence per summary)
            for word in top_features:
                if word in s:
                    pair_count[occupation][word] += 1
            
            # Count co-occurrences: deduplicate within each summary
            features_in_summary = [word for word in s if word in top_features]
            in_this_summary = set()
            for i in range(len(features_in_summary)):
                for j in range(i + 1, len(features_in_summary)):
                    # Ensure each pair is counted only once per summary
                    if (features_in_summary[i], features_in_summary[j]) not in in_this_summary:
                        in_this_summary.add((features_in_summary[i], features_in_summary[j]))
                        feature_counts[features_in_summary[i]] += 1
                        co_occurrence_counts[features_in_summary[i]][features_in_summary[j]] += 1
        
        # Build probability matrix from co-occurrence counts
        matrix_size = len(top_features)
        probability_matrix = np.zeros((matrix_size, matrix_size))
        
        for i, f1 in enumerate(top_features):
            total_co_occurrences = sum(co_occurrence_counts[f1].values())
            for j, f2 in enumerate(top_features):
                if total_co_occurrences > 0:
                    probability_matrix[i, j] = co_occurrence_counts[f1][f2] / total_co_occurrences
        
        all_matrices[occupation] = {'features': top_features, 'matrix': probability_matrix}
        
        # Calculate fraction of summaries that contain each top feature
        for word in top_features:
            feature_probs[occupation][word] = pair_count[occupation][word] / occ_count[occupation]
    
    return feature_probs, all_matrices

def plot_heatmap(data, occupation):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Unpack the data
    features = data['features']
    matrix = data['matrix']

    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 10))  # Increase figure size for better readability
    sns.heatmap(matrix, xticklabels=features, yticklabels=features, 
                cmap='coolwarm', center=0, linewidths=0.5, cbar_kws={'label': 'Co-occurrence Probability'})

    plt.title(f'Feature Co-occurrence Heatmap for {occupation}', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.show()

# Save feature probabilities (already JSON-serializable)
def save_feature_probs(feature_probs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_probs, f, indent=2, ensure_ascii=False)

# Save all_matrices by converting NumPy arrays to lists
def save_all_matrices(all_matrices, output_path):
    all_matrices_serializable = {}
    for occ, mat_dict in all_matrices.items():
        all_matrices_serializable[occ] = {
            "features": mat_dict["features"],
            "matrix": mat_dict["matrix"].tolist()  # convert NumPy array to list
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_matrices_serializable, f, indent=2, ensure_ascii=False)