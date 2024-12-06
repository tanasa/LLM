#!/usr/bin/env python
# coding: utf-8

# In[5]:


# PROBLEM 2


# In[6]:


def skip_ngrams(text, max_window_size=3):
    
    """
    Generate skip-grams for the given text within the specified window sizes.

    Parameters:
    - text (str): The input text to generate skip-grams from.
    - max_window_size (int): The maximum window size to consider for skip-grams.

    Returns:
    - List of tuples containing (input_word, label_word).
    """
    
    # Step 1 : Split the text into separate words
    words = text.split()
    skip_grams = []

    # Step 2 : Iterate over each word with its index
    
    for index, word in enumerate(words):
    
        # Create pairs with neighboring words within the window sizes
        
        for window_size in range(1, max_window_size + 1):
            # Locate and get the left neighbor
            if index - window_size >= 0:
                skip_grams.append((word, words[index - window_size]))
            # Locate and get the right neighbor
            if index + window_size < len(words):
                skip_grams.append((word, words[index + window_size]))

    return skip_grams

if __name__ == "__main__":

    text = "data science professionals have promising career path"
    window_sizes = [1, 2, 3]

    for window in window_sizes:
        print(f"Window Size = {window}")
        skip_grams = skip_ngrams(text, max_window_size=window)
        
        # Print the header
        print(f"Index \t Input \t Label")
       
        # Print each skip-gram pair with its index
        for i, (input_word, label_word) in enumerate(skip_grams):
            print(f"{i} \t {input_word} \t {label_word}")
        
        # Print the total number of entries
        total_entries = len(skip_grams)
        print(f"\nNumber of skip-gram entries for window size {window}: {total_entries}")


# In[7]:


# PROBLEM 3


# In[8]:


import numpy as np

def skip_ngrams(text, max_window_size=2, structured_output=False):
    """
    Generate skip-grams for the given text within the specified window sizes.

    Parameters:
    - text (str): The input text to generate skip-grams from.
    - max_window_size (int): The maximum window size to consider for skip-grams.
    - structured_output (bool): If True, return a list of dictionaries instead of tuples.

    Returns:
    - List of tuples or dictionaries containing (input_word, label_word).
    """
    
    # Step 1: Split the text into separate words
    
    words = text.split()
    skip_grams = []

    # Step 2: Iterate over each word with its index
    
    for index, word in enumerate(words):
    
        # Create pairs with neighboring words within the window sizes
        for window_size in range(1, max_window_size + 1):
            
            # Locate and get the left neighbor
            if index - window_size >= 0:
                if structured_output:
                    skip_grams.append({"input_word": word, "label_word": words[index - window_size]})
                else:
                    skip_grams.append((word, words[index - window_size]))

            # Locate and get the right neighbor
            if index + window_size < len(words):
                if structured_output:
                    skip_grams.append({"input_word": word, "label_word": words[index + window_size]})
                else:
                    skip_grams.append((word, words[index + window_size]))

    return skip_grams

def one_hot_encode(skip_grams, unique_words):
    """
    Generate one-hot encoded input and output data from skip-grams.

    Parameters:
    - skip_grams (list): The list of skip-grams.
    - unique_words (list): The list of unique words in the text.

    Returns:
    - Tuple of numpy arrays: (input_hot, output_hot)
    """
    
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}
    
    input_hot = np.zeros((len(skip_grams), len(unique_words)), dtype=int)
    output_hot = np.zeros((len(skip_grams), len(unique_words)), dtype=int)

    for i, (input_word, label_word) in enumerate(skip_grams):
        input_index = word_to_index[input_word]
        label_index = word_to_index[label_word]

        input_hot[i, input_index] = 1
        output_hot[i, label_index] = 1

    return input_hot, output_hot, word_to_index

if __name__ == "__main__":
    text = "data science professionals have promising career path"
    window_size = 2

    # Generate skip-grams
    skip_grams = skip_ngrams(text, max_window_size=window_size)

    # Define unique words in the desired order
    unique_words = ['career', 'data', 'have', 'path', 'professionals', 'promising', 'science']

    print(f"Window Size = {window_size}")
    
    print(f"Index \t Input \t Label")
    for i, (input_word, label_word) in enumerate(skip_grams):
            print(f"{i} \t {input_word} \t {label_word}")
        
    # Print the total number of entries
    total_entries = len(skip_grams)
    print(f"\nNumber of skip-gram entries for window size {window}: {total_entries}")
    
    # Generate one-hot encoded data
    input_hot, output_hot, word_to_index = one_hot_encode(skip_grams, unique_words)


    print("Unique Words:", unique_words)
    print("\nInput One-Hot Encoded:")
    print(input_hot)
    print("\nOutput One-Hot Encoded:")
    print(output_hot)

    # Print the mapping of words to their indices
    print("\nWord to Index Mapping:")
    for word in unique_words:
        print(f"{word}: {word_to_index[word]}")


# In[ ]:




