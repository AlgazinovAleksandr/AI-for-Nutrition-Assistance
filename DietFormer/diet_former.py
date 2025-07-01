from autogen import AssistantAgent, UserProxyAgent
from constants import api_key, model_name, base_url
import re

llm_config = {"config_list": [{
            "model": model_name,  # or other supported model IDs
            "base_url": base_url,
            "api_key": api_key,
        }]}

system_prompt_category = """You are DietIdentifierBot, an assistant whose sole job is to read a user’s free‑form diet‑related prompt and map it to exactly one of the following nutritional objectives. After picking the one best class, you must output that class name followed by the exact token “TERMINATE” (nothing else).

Your 16 classes:
1. Weight loss / fat loss  
   • Shed body fat via calorie deficit  
2. Muscle gain / hypertrophy  
   • Build lean muscle via calorie surplus + protein focus  
3. Weight maintenance  
   • Stay at current weight, balance calories in = calories out  
4. Body recomposition  
   • Simultaneously lose fat and gain muscle  
5. Bulking  
   • Aggressive muscle‑building phase (calorie surplus > maintenance)  
6. Cutting  
   • Aggressive fat‑loss phase (calorie deficit while preserving muscle)  
7. Lowering blood pressure  
   • Dietary changes to reduce hypertension  
8. Improving blood sugar control / insulin sensitivity  
   • Manage glucose levels (e.g. for prediabetes/diabetes)  
9. Lowering cholesterol / improving heart health  
   • Reduce LDL, boost heart‑healthy fats  
10. Improving gut health  
    • Support microbiome via fiber, pre‑/probiotics  
11. Boosting immune function  
    • Nutrient focus to strengthen immunity  
12. Enhancing energy levels  
    • Dietary tweaks to sustain all‑day energy  
13. Reducing sugar & processed‑food intake  
    • Cut back on sweets and ultra‑processed items  
14. Adopting a specific dietary pattern  
    • e.g. vegetarian, vegan, Mediterranean, ketogenic  
15. Time‑restricted eating / intermittent fasting  
    • Eating only within a defined daily window  
16. UNK  
    • The user’s request is outside the scope of diet/nutrition objectives or is nonsensical in this context  

FORMAT RULES  
• Only output:  
  `<Class Name>` TERMINATE  
  (with no additional text, explanation, or formatting)  

FEW‑SHOT EXAMPLES

User: “I want to drop about 10 pounds by cutting calories but keep my muscle definition.”  
Bot: Weight loss / fat loss TERMINATE

User: “I’m looking to bulk up for the rugby season—need to be eating way above maintenance with tons of protein.”  
Bot: Bulking TERMINATE

User: “Let’s focus on cleaning up my diet so I can manage my blood sugar—maybe fewer carbs and more fiber.”  
Bot: Improving blood sugar control / insulin sensitivity TERMINATE

User: “I’m happy with my weight but want to mix in strength gains while trimming fat.”  
Bot: Body recomposition TERMINATE

User: “What kind of meal plan would help me keep my current weight steady?”  
Bot: Weight maintenance TERMINATE

User: “Please suggest a 16:8 eating schedule to help me fast for 16 hours and eat in an 8-hour window.”  
Bot: Time‑restricted eating / intermittent fasting TERMINATE

User: “How can I add more probiotics and prebiotics to my meals to improve digestion?”  
Bot: Improving gut health TERMINATE

User: “Tell me about quantum physics.”  
Bot: UNK TERMINATE
"""

# Create the agent with the defined system prompt
nutritional_assistant = AssistantAgent(
    name="Nutritional_Assistant",
    system_message=system_prompt_category,
    llm_config=llm_config, # You can specify your desired LLM model here
)

user_proxy_nutritional = UserProxyAgent(
        name="User",
        human_input_mode="NEVER", # Set to "ALWAYS" to allow human input
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )

def parse_measurement(s: str):
    m = re.match(r'^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)?\s*$', s)
    if not m:
        return None, None
    return float(m.group(1)), (m.group(2) or '').lower()

def to_kg(value, unit):
    unit = unit.lower().strip().rstrip('.')
    kg_units = {
        '', 'kg', 'kgs', 'kilogram', 'kilograms',
        'kilo', 'kilos', 'k', 'kilogramme', 'kilogrammes',
        'kil', 'kgr', 'kgrs'
    }
    lb_units = {
        'lb', 'lbs', 'pound', 'pounds',
        'pd', 'pds', 'p', 'l', 'pnd', 'pnds'
    }
    if unit in kg_units:
        return value
    if unit in lb_units:
        return value * 0.45359237
    raise ValueError(f"Unknown weight unit “{unit}”")

def to_cm(value, unit):
    # Normalize: lowercase, strip whitespace, remove trailing dots and quotes
    unit = unit.lower().strip().rstrip('.').replace('"', '').replace("'", '').strip()
    cm_units = {
        '', 'cm', 'cms', 'centimeter', 'centimeters',
        'centimetre', 'centimetres', 'centi', 'c'
    }
    m_units = {
        'm', 'mt', 'meter', 'meters',
        'metre', 'metres'
    }
    in_units = {
        'in', 'inch', 'inches', 'ins', 'inchs'
    }
    ft_units = {
        'ft', 'feet', 'foot', 'fts'
    }
    if unit in cm_units:
        return value
    if unit in m_units:
        return value * 100
    if unit in in_units:
        return value * 2.54
    if unit in ft_units:
        return value * 30.48
    raise ValueError(f"Unknown height unit “{unit}”")

def prompt_measurement(field, default_unit, converter, example_value):
    raw = input(f"{field.capitalize()} (e.g. “{example_value} {default_unit}” or with other units), or press Enter to skip: ").strip()
    if not raw:
        return None
    val, unit = parse_measurement(raw)
    if val is None:
        print("❗ Could not parse—try again.")
        return prompt_measurement(field, default_unit, converter, example_value)
    try:
        return converter(val, unit)
    except ValueError as e:
        print(f"❗ {e}")
        return prompt_measurement(field, default_unit, converter, example_value)

def normalize_gender(s: str) -> str:
    """Normalize gender input to 'male' or 'female', else None."""
    s_clean = s.strip().lower()
    if s_clean in ('m', 'male', 'man'):
        return 'male'
    if s_clean in ('f', 'female', 'woman'):
        return 'female'
    return None

def prompt_gender():
    raw = input("Gender (m / male or f / female), or press Enter to skip: ").strip()
    if not raw:
        return None
    gender = normalize_gender(raw)
    if gender is None:
        print("❗ Unrecognized input. Please enter 'm' or 'f'.")
        return prompt_gender()
    return gender

def prompt_choice(field, choices):
    print(f"{field.capitalize()}—choose one (press Enter to skip):")
    for i, opt in enumerate(choices, 1):
        print(f"  {i}. {opt}")
    raw = input("Enter number or text: ").strip()
    if not raw:
        return None
    # try number
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    # try text match
    for opt in choices:
        if opt.lower() == raw.lower():
            return opt
    print("❗ Invalid choice—please try again.")
    return prompt_choice(field, choices)

def prompt_text(field, default_note):
    raw = input(f"{field.capitalize()} ({default_note}), or press Enter to skip: ").strip()
    return raw or None

def generate_summary(age, gender, weight_kg, height_cm, activity, bmi, conditions, allergies, diet_pref, meal_pattern):
    summary_parts = []

    if age:
        summary_parts.append(f"The user is {age} years old")
    else:
        summary_parts.append("The user's age is unknown")

    if gender:
        summary_parts.append(f"{'a male' if gender == 'male' else 'a female'}")
    else:
        summary_parts.append("with unspecified gender")

    if weight_kg:
        summary_parts.append(f"weighs {weight_kg:.1f} kg")
    if height_cm:
        summary_parts.append(f"and is {height_cm:.1f} cm tall")

    if activity:
        summary_parts.append(f"Their physical activity level is: {activity.lower()}.")

    if bmi:
        summary_parts.append(f"The calculated BMI is {bmi:.1f}.")

    if conditions:
        summary_parts.append(f"They have the following medical condition(s): {conditions}.")
    else:
        summary_parts.append("They have no known medical conditions.")

    if allergies:
        summary_parts.append(f"Reported food allergies or intolerances: {allergies}.")
    else:
        summary_parts.append("No food allergies or intolerances were reported.")

    if diet_pref:
        summary_parts.append(f"Dietary preferences or restrictions: {diet_pref}.")
    else:
        summary_parts.append("No dietary restrictions or preferences.")

    if meal_pattern:
        summary_parts.append(f"Their typical meal pattern includes: {meal_pattern}.")
    else:
        summary_parts.append("Meal pattern was not specified.")

    return ' '.join(summary_parts)

def ask_for_parameters():
    weight_kg    = prompt_measurement('weight', 'kg',   to_kg, example_value=70)
    print(f"Weight:   {weight_kg:.1f} kg"    if weight_kg    else "Weight:   (skipped)")
    height_cm    = prompt_measurement('height', 'cm',   to_cm, example_value=170)
    print(f"Height:   {height_cm:.1f} cm"    if height_cm    else "Height:   (skipped)")
    age          = prompt_text('age (years)', 'e.g. 30')
    print(f"Age:      {age} years"           if age          else "Age:      (skipped)")
    gender       = prompt_gender()
    print(f"Gender:   {gender}"              if gender       else "Gender:   (skipped)")
    activity_levels = [
        'Sedentary (little/no exercise)',
        'Lightly active (1–3 days/week)',
        'Moderately active (3–5 days/week)',
        'Very active (6–7 days/week)',
        'Extra active (hard exercise & physical job)'
    ]
    activity     = prompt_choice('activity level', activity_levels)
    print(f"Activity: {activity}"            if activity     else "Activity: (skipped)")
    conditions   = prompt_text('medical conditions', 'e.g. none')
    print(f"Medical conditions: {conditions}"   if conditions   else "Medical conditions: none")
    allergies    = prompt_text('food allergies/intolerances', 'e.g. none')
    print(f"Allergies/intolerances: {allergies}" if allergies    else "Allergies/intolerances: none")
    diet_pref    = prompt_text('dietary preferences/restrictions', 'e.g. none')
    print(f"Diet preferences: {diet_pref}"      if diet_pref    else "Diet preferences: none")
    meal_pattern = prompt_text('typical meal pattern (meals & snacks per day)', 'e.g. 3 meals + 2 snacks')
    print(f"Meal pattern: {meal_pattern}"       if meal_pattern else "Meal pattern: (skipped)")

    # Optionally compute BMI if we have weight & height
    if weight_kg and height_cm:
        h_m = height_cm / 100
        bmi = weight_kg / (h_m*h_m)
        print(f"Calculated BMI: {bmi:.1f}")
    parameters = generate_summary(age, gender, weight_kg, height_cm, activity, bmi, conditions, allergies, diet_pref, meal_pattern)
    return parameters

# 1) Define the system prompt
system_prompt_diet = """
You are DietFormer, an expert diet‐formulation agent.

Your job is to generate a personalized diet plan for a user based on:
  • `goals`: a plain-text objective such as "Weight loss / fat loss", chosen from one of 15 predefined categories.
  • `parameters`: plain-text description of the user's profile, including age, sex, weight, height, activity level, BMI, medical conditions, allergies, dietary restrictions, and meal habits.

Your response must follow these instructions **exactly**:

1. Output only a single string variable named `descibing_the_diet`.
   - This string must contain the entire diet plan.
   - It should be a raw string with no formatting around the variable name.
   - Do NOT use Markdown formatting like `**` or backticks.

2. The diet plan must:
   - Align with the user’s `goals`
   - Respect medical conditions, allergies, and food preferences
   - Include a daily calorie target and macronutrient breakdown
   - Provide a sample day of eating (breakfast, lunch, dinner, snacks)
   - Include timing suggestions or fasting windows if appropriate
   - Be clear, practical, and easy to follow

3. ✅ **You MUST end your output with the exact word**:

TERMINATE

This word must:
- Appear **alone** on the final line
- Have **no formatting** (do NOT bold it, quote it, or wrap it)
- Have **no leading or trailing whitespace**
- Not be surrounded by punctuation or any other symbols
- Be the **last and only thing** after the diet string

❌ Wrong: `**TERMINATE**`  
❌ Wrong: `TERMINATE.`  
❌ Wrong: `"TERMINATE"`  
❌ Wrong: `TERMINATE  `  
✅ Correct:  
TERMINATE

Example final output format (the last 2 lines should look exactly like this):

descibing_the_diet = [your full diet plan here]  
TERMINATE
"""

# Create the agent with the defined system prompt
diet_former = AssistantAgent(
    name="Diet_Former",
    system_message=system_prompt_diet,
    llm_config=llm_config, # You can specify your desired LLM model here
)

user_proxy_diet = UserProxyAgent(
        name="User",
        human_input_mode="NEVER", # Set to "ALWAYS" to allow human input
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )

def run_the_diet_former(output_path="diet_result.txt"):
    message_nutrition = input('Enter your nutritional characteristics and objectives : ')
    chat_result = user_proxy_nutritional.initiate_chat(
    nutritional_assistant,
    message=message_nutrition
)
    # Extract the latest message (usually the answer)
    goals = chat_result.chat_history[-1]['content'].split('TERMINATE')[0].strip()
    if goals == 'UNK':
        print('I did not understand your request, please try again or contact me next time')
        return
    parameters = ask_for_parameters()
    message_diet = f"The user goal is: {goals}; The user parameters are: {parameters}; Please formulate a personalized diet plan based on the user's goal and parameters."
    diet_result = user_proxy_diet.initiate_chat(
    diet_former,
    message=message_diet
    )
    # Extract the latest message (usually the answer)
    diet_result = diet_result.chat_history[-1]['content'].split("TERMINATE")[0]
    with open(output_path, "w") as f:
        f.write(diet_result)
    return diet_result