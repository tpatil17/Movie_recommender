
import pandas as pd
import ast
import numpy as np


# Load the core files from drive to make them ready for pre processing

# Load the core files we need for the Hybrid system
# low_memory=False is needed because some columns have mixed types (IDs)
def load_clean_data():
    data_dir: str = "data/raw"
    meta = pd.read_csv(f'{data_dir}/movies_metadata.csv', low_memory=False)
    ratings = pd.read_csv(f'{data_dir}/ratings_small.csv') # We'll start with small ratings for testing
    credits = pd.read_csv(f'{data_dir}/credits.csv')
    keywords = pd.read_csv(f'{data_dir}/keywords.csv')

    # Data Cleaning
    # Movies_metadata has a messy Id's columns so needs cleaning before merging
    meta['id'] = pd.to_numeric(meta['id'], errors='coerce') # Turn bad IDs into NaN
    meta = meta.dropna(subset=['id']) # Drop the bad rows
    meta['id'] = meta['id'].astype(int) # Convert to integer

    # Credits
    # To make sure both tables have the same type of ID's
    credits['id'] = credits['id'].astype(int)
    # Merge Movies Metadata and Credits on ID's
    data = meta.merge(credits, on='id')
    # Cast and crew are now on the main data frame
    return meta, ratings, credits, keywords, data

# Parsing JSON
# Feature Engineering cell
# Columns Cast, Crew and Genre contain Json data
def parse_features(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def apply_parse_features(data):
    features = ['cast', 'crew', 'genres']
    for feature in features:
        data[feature] = data[feature].apply(parse_features) # for each feature (column) apply the parse Json function
    return data


# extract features from the newly modified data
# extract Director names
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def apply_get_director(data):
    data['director'] = data['crew'].apply(get_director)
    return data


#Extract top three crew/cast of the movie

def get_top_3_cast(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def apply_get_top_3_cast(data):
    data['top_cast'] = data['cast'].apply(get_top_3_cast)
    return data

# Extracting Genre's
def extract_list_names(x):
    if isinstance(x, list):
        # Loop through the list and pick out the 'name' key
        return [i['name'] for i in x]
    return []

# Apply it to the genres column
def apply_extract_list_names(data):
    data['genres'] = data['genres'].apply(extract_list_names)
    return data




# Cleaning the text data
# we convert all the text into lower case and get rid of white space
# this process is aimed to simplify data and make sure the model can distinguis between similar yet different text
def clean_data(x):
    if isinstance(x, list):
        # Convert list of strings to lowercase and remove spaces
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply to features
def apply_clean_data(data):
    features = ['top_cast', 'genres', 'director']
    for feature in features:
        data[feature] = data[feature].apply(clean_data)
    return data


# Creating a Meta data soup by joining the features into a soup
def create_soup(x):
    return ' '.join(x['top_cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def apply_create_soup(data):    
    data['soup'] = data.apply(create_soup, axis=1)
    return data




