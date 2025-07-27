import pandas as pd
from food_extractor import extract_food_items, prompt_llm_for_ingredients, get_basic_ingredient_names, get_composed_dish_names, LLMDBMatcherAgent

def get_final_weight(item, quantity, basic_ingredients_db):
    # If the LLM returns a string like 'reference value', use the DB reference weight
    if isinstance(quantity, str) and 'reference' in quantity.lower():
        row = basic_ingredients_db[basic_ingredients_db['Food name'].str.lower() == item.lower()]
        if not row.empty:
            ref_weight = row.iloc[0]['Weight ']
            try:
                return float(ref_weight)
            except Exception:
                return 100.0  # fallback
        return 100.0  # fallback
    try:
        return float(quantity)
    except Exception:
        return 100.0  # fallback

# Load DB and filter for basic ingredients
raw_db = pd.read_excel('Food_all+General_items_with_ingredients.xlsx')
raw_db.columns = raw_db.iloc[0]
db = raw_db[1:]
basic_ingredients_db = db[db['Ingredient'] == 1]



def get_nutrition_for_ingredient(ingredient_name, weight, db, search_composed=False):
    if search_composed:
        search_db = db[db['Ingredient'] == 0]
        threshold = 0.95
    else:
        search_db = db[db['Ingredient'] == 1]
        threshold = 0.75

    best_score = 0
    best_row = None
    best_name = None
    for _, row in search_db.iterrows():
        db_name = str(row['Food name'])
        score = 1.0 if db_name.lower() == ingredient_name.lower() else 0.0
        if score > best_score:
            best_score = score
            best_row = row
            best_name = db_name
    if best_row is not None:
        reference_weight = float(best_row['Weight ']) if 'Weight ' in best_row and pd.notnull(best_row['Weight ']) else 100.0
        weight_ratio = weight / reference_weight if reference_weight > 0 else 1.0
        print(f"      [Nutrition Calculation] For '{ingredient_name}':")
        print(f"         - Used weight: {weight}g")
        print(f"         - DB reference weight: {reference_weight}g")
        print(f"         - Weight ratio: {weight_ratio:.3f}")
        def safe_nutrition_value(value):
            try:
                numeric_value = pd.to_numeric(value, errors='coerce')
                if numeric_value != numeric_value or numeric_value == 0:
                    return 0
                return round(numeric_value * weight_ratio, 1)
            except (ValueError, TypeError):
                return 0
        nutrition = {
            'calories': safe_nutrition_value(best_row['Calories(kcal)']),
            'protein': safe_nutrition_value(best_row['Protein']),
            'fat': safe_nutrition_value(best_row['Total Fat']),
            'carbs': safe_nutrition_value(best_row['Carbohydrate']),
            'fiber': safe_nutrition_value(best_row['Total Dietary Fibre']),
            'sugar': safe_nutrition_value(best_row['Total Sugar']),
            'sodium': safe_nutrition_value(best_row['Sodium']),
        }
        print(f"         - Nutrition: {nutrition}")
        return nutrition
    print(f"❌ No good match found for ingredient '{ingredient_name}'. Best match: '{best_name}' (score: {best_score:.2f})")
    return None

def sum_nutrition(nutrition_list):
    total = {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0, 'fiber': 0, 'sugar': 0, 'sodium': 0}
    for n in nutrition_list:
        if n:
            for k in total:
                total[k] += n.get(k, 0)
    return total

user_input = input("What did you eat? ")
food_items = extract_food_items(user_input)

if isinstance(food_items, dict) and 'error' in food_items:
    print(f"❌ Error extracting food items: {food_items['error']}")
    if 'raw_response' in food_items:
        print(f"Raw LLM response: {food_items['raw_response']}")
    exit(1)

print(f"✅ Extracted food items: {food_items}")

total_nutrition = {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0, 'fiber': 0, 'sugar': 0, 'sodium': 0}

# Create agents for composed dishes and basic ingredients
composed_dish_names = get_composed_dish_names(db)
basic_ingredient_names = get_basic_ingredient_names(db)
composed_agent = LLMDBMatcherAgent(composed_dish_names, agent_name="ComposedDBMatcher", exact_match_only=True)
basic_agent = LLMDBMatcherAgent(basic_ingredient_names, agent_name="BasicDBMatcher")

for item, quantity in food_items.items():
    print(f"\n🔍 Processing: {item} ({quantity})")
    # First, try to match the composed dish using the original item name
    composed_match = composed_agent.match(item)
    if composed_match:
        # Always use the reference weight for the matched composed dish
        row = db[(db['Food name'].str.lower() == composed_match.lower()) & (db['Ingredient'] == 0)]
        if not row.empty:
            weight = float(row.iloc[0]['Weight '])
        else:
            weight = 100.0  # fallback
        print(f"   LLM composed dish match: '{composed_match}'")
        nutrition = get_nutrition_for_ingredient(composed_match, weight, db, search_composed=True)
        print(f"   Nutrition (composed): {nutrition}")
        for k in total_nutrition:
            total_nutrition[k] += nutrition[k]
    else:
        # If not found, decompose and search basic ingredients
        print(f"   ➡️ '{item}' is a composite dish. Decomposing...")
        ingredients = prompt_llm_for_ingredients(item)
        print(f"   Decomposed ingredients: {ingredients}")
        nutrition_list = []
        for ingr, ingr_amt in ingredients.items():
            ingr_weight = get_final_weight(ingr, ingr_amt, basic_ingredients_db)
            basic_match = basic_agent.match(ingr)
            print(f"      - '{ingr}': LLM best: '{basic_match}'")
            used_name = basic_match
            print(f"        → Using LLM match for nutrition lookup.")
            ingr_nutrition = get_nutrition_for_ingredient(used_name, ingr_weight, db, search_composed=False)
            print(f"        Nutrition: {ingr_nutrition}")
            nutrition_list.append(ingr_nutrition)
        summed = sum_nutrition(nutrition_list)
        for k in total_nutrition:
            total_nutrition[k] += summed[k]

print(f"\n📊 TOTAL NUTRITION VALUES:")
print(f"   Calories: {total_nutrition['calories']:.1f} kcal")
print(f"   Protein: {total_nutrition['protein']:.1f}g")
print(f"   Fat: {total_nutrition['fat']:.1f}g")
print(f"   Carbs: {total_nutrition['carbs']:.1f}g")
print(f"   Fiber: {total_nutrition['fiber']:.1f}g")
print(f"   Sugar: {total_nutrition['sugar']:.1f}g")
print(f"   Sodium: {total_nutrition['sodium']:.1f}mg")
    
    
