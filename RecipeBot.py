import streamlit as st
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):  
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()

ingredient_list = load_ingredient_data()

# ✅ Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# ✅ Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  # ✅ Uses angular distance (1 - cosine similarity)
    
    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)
    
    index.build(50)  # ✅ More trees = better accuracy

    return index
annoy_index = build_annoy_index()

# ✅ Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

# def direct_search_alternatives(ingredient):
#     # Convert the input ingredient to a vector
#     ingredient_vec = nlp(ingredient.lower()).vector
    
#     # Compute cosine similarity with all vectors in ingredient_vectors
#     similarities = [cosine_similarity(ingredient_vec, vec) for vec in ingredient_vectors]
    
#     # Get the indices of the most similar ingredients (sorted in descending order)
#     top_indices = np.argsort(similarities)[::-1]
    
#     # Retrieve alternatives, excluding the input ingredient itself
#     alternatives = [filtered_ingredient_list[i] for i in top_indices if filtered_ingredient_list[i].lower() != ingredient.lower()]
    
#     # Return the top 3 alternatives (excluding the original ingredient)
#     return alternatives[:3]

# ✅ Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

def direct_search_alternatives(ingredient):
    """
    Step 0: Locate the function direct_search_alternatives(ingredient).
    Step 1: Retrieve the vector embedding for the input ingredient.
    Step 2: Iterate through all ingredients in ingredient_list and compute cosine similarity.
    Step 3: Sort the ingredients by similarity score in descending order.
    Step 4: Return the top-3 most similar ingredients, excluding the input itself.
    """

    # Step 1: Convert the input ingredient to a vector
    ingredient_vec = nlp(ingredient.lower()).vector

    # Step 2: Iterate through all ingredients in ingredient_list and compute cosine similarity
    similarities = []
    for vec in ingredient_vectors:
        similarities.append(cosine_similarity(ingredient_vec, vec))

    # Step 3: Sort the ingredients by similarity score in descending order
    sorted_indices = np.argsort(similarities)[::-1]  # Highest similarity first

    # Step 4: Retrieve the top-3 most similar ingredients (excluding the input itself)
    alternatives = [
        filtered_ingredient_list[i] for i in sorted_indices 
        if filtered_ingredient_list[i].lower() != ingredient.lower()
    ]

    return alternatives[:3]  # Return the top 3 alternatives


# ✅ Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient):
    # Convert the input ingredient to a vector
    ingredient_vec = nlp(ingredient.lower()).vector
    
    # Perform the Annoy search to find more than 3 nearest neighbors
    neighbor_ids = annoy_index.get_nns_by_vector(ingredient_vec, n=5)  # Get extra candidates in case of duplicates

    # Retrieve corresponding ingredient names using the neighbor IDs
    alternatives = [filtered_ingredient_list[i] for i in neighbor_ids]

    # Remove the input ingredient if it's in the results
    alternatives = [alt for alt in alternatives if alt.lower() != ingredient.lower()]

    # Return only the top 3 alternatives after filtering
    return alternatives[:3]


# ✅ Generate Recipe
def generate_recipe(ingredients, cuisine):
    # orignal format
    # input_text = (
    #     f"Ingredients: {', '.join(ingredients.split(', '))}\n"
    #     f"Cuisine: {cuisine}\n"
    #     f"Let's create a dish inspired by {cuisine} cuisine with these ingredients. Here are the preparation and cooking instructions:"
    # ) 

    # Control Response Format  
    input_text = (
        f"Title: write a recipe name\n"
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
        f"Let's create a dish inspired by {cuisine} cuisine with these ingredients. Here are the recipe title, list of ingredients and step by step cooking instructions:"  
    ) 

    # Adjust Detail Level (Concise)
    # input_text = (
        
    #     f"Ingredients: {', '.join(ingredients.split(', '))}\n"
    #     f"Cuisine: {cuisine}\n"
    #     f"Let's create a dish inspired by {cuisine} cuisine and share recipe with brief instruction:"  
    # ) 

    # Adjust Detail Level (Detailed)
    # input_text = (
    #     f"Title: write a recipe name\n"
    #     f"Ingredients: {', '.join(ingredients.split(', '))}\n"
    #     f"Cuisine: {cuisine}\n" 
    #     f"Let's create a dish inspired by {cuisine} cuisine with recipe with detailed instruction:"  
    # ) 

    #Encourage creativity 
    # input_text = (
    #     f"Ingredients: {', '.join(ingredients.split(', '))}\n"
    #     f"Cuisine: {cuisine}\n" 
    #     f"Let's create a dish inspired by {cuisine} cuisine. Surprise users with unique combinations:"  
    # )     
    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                             max_length=250, num_return_sequences=1,
                             repetition_penalty=1.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

# ✅ Measure Search Time
import time

# ingredient = "zwieback toast crumb"

# # Measure Direct Search Time
# start_time = time.time()
# direct_search_alternatives(ingredient)
# print("Direct Search Time:", time.time() - start_time)

# # Measure Annoy Search Time
# start_time = time.time()
# annoy_search_alternatives(ingredient)
# print("Annoy Search Time:", time.time() - start_time)

# ✅ Generate Health Tip
def generate_health_tip(nutrition_info):
    # Create the input prompt for the health model
    input_text = (
        f"Given the following nutritional information from a recipe:\n"
        f"Calories: {nutrition_info['calories']} kcal\n"
        f"Protein: {nutrition_info['protein']} g\n"
        f"Carbs: {nutrition_info['carbs']} g\n"
        f"Fat: {nutrition_info['fat']} g\n"
        f"Let's write a health tip for this recipe to improve its nutritional balance or suggest healthier options:"
    )

    # Generate the health tip with the model
    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                             max_length=250, num_return_sequences=1, 
                             repetition_penalty=1.2, temperature=0.8, top_k=50, top_p=0.95, do_sample=True)
    
    # Return the generated health tip
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


import requests

# Define the function to fetch nutritional data from USDA or another source
def fetch_nutrition_data(ingredient):
    # Example: Use the USDA API or another source. The URL and key are placeholders.
    api_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={ingredient}&api_key=pkeNFiTvuuULKI1hYl8rrJAX73N1maDdieADSABZ"
    
    # Call the API
    response = requests.get(api_url)
    data = response.json()
    
    # Check if there are results and return nutritional data
    if data['foods']:
        food = data['foods'][0]
        nutrition = {
            'calories': food['foodNutrients'][3]['value'],  # calories
            'protein': food['foodNutrients'][0]['value'],   # protein
            'carbs': food['foodNutrients'][1]['value'],     # carbs
            'fat': food['foodNutrients'][2]['value'],       # fat
        }
        return nutrition
    else:
        return None

# Function to calculate and return the health analysis of the recipe
def health_analysis(ingredients):
    total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    
    # Loop through ingredients and fetch their nutrition
    for ingredient in ingredients.split(', '):
        nutrition = fetch_nutrition_data(ingredient)
        if nutrition:
            total_nutrition['calories'] += nutrition['calories']
            total_nutrition['protein'] += nutrition['protein']
            total_nutrition['carbs'] += nutrition['carbs']
            total_nutrition['fat'] += nutrition['fat']
    
    return total_nutrition

# ✅ Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])
if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["recipe"] = generate_recipe(ingredients, cuisine)

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(label="📂 Save Recipe", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")
    
    # Get nutritional info for the ingredients
    if ingredients:
        nutrition_info = health_analysis(ingredients)
        
        # Display nutritional information
        st.markdown(f"### 🍽️ Nutritional Breakdown for Recipe Ingredients:")
        st.markdown(f"**Calories**: {nutrition_info['calories']} kcal")
        st.markdown(f"**Protein**: {nutrition_info['protein']} g")
        st.markdown(f"**Carbohydrates**: {nutrition_info['carbs']} g")
        st.markdown(f"**Fat**: {nutrition_info['fat']} g")

        # Generate and display a health tip based on the nutritional info
        st.markdown("### 🌿 Health Tip:")
        health_tip = generate_health_tip(nutrition_info)
        st.markdown(f"**Tip**: {health_tip}")
    
    # ✅ Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
    
        # Health analysis section
    st.markdown("---")
    st.markdown("## 🍏 Health Analysis of Ingredients")