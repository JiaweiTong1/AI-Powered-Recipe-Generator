# AI-Powered Recipe Generator
This project is an AI-powered recipe generator that allows users to input ingredients and generate recipes, analyze nutritional information, and find alternative ingredients. The app is built using Streamlit and leverages machine learning models for natural language processing and recommendation tasks.

## Features
- Generate recipes based on user-provided ingredients and cuisine type.
- Analyze the nutritional breakdown of the recipe ingredients.
- Provide health tips based on the nutritional analysis.
- Suggest alternative ingredients using Annoy and cosine similarity.

## Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Powered-Recipe-Generator.git
   cd AI-Powered-Recipe-Generator
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r Requirements.txt
   ```

4. Download the ingredient data:
   The app will automatically download the required ingredient data (`processed_ingredients_with_id.csv`) from Google Drive when you run it for the first time.

5. Start the Streamlit app:
   ```bash
   streamlit run RecipeBot.py
   ```

## Notes
- The app uses the `en_core_web_lg` SpaCy model, which will be downloaded automatically if not already installed.
- Ensure you have an active internet connection for downloading the ingredient data and SpaCy model.