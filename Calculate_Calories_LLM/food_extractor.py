import json
import re
import ast
from autogen import AssistantAgent, UserProxyAgent
from constants import zhipu_api_key, zhipu_base_url, default_model

def extract_food_items(user_input):
    """Extract food items and their estimated weights in grams from user input using Zhipu AI. If no quantity is specified, return 'reference value'."""
    llm_config = {"config_list": [{
        "model": "glm-3-turbo",  # Explicitly set model name
        "base_url": zhipu_base_url,
        "api_key": zhipu_api_key,
    }]}
    print(f"🔧 Using model: {llm_config['config_list'][0]['model']}")
    # System prompt for food extraction
    system_prompt = (
        "Extract all food items and their estimated weights in grams from the user's input. "
        "Translate all food item names you extract into English before returning them. "
        "Return ONLY a JSON object in the format: {\"food_items\": {\"item1\": quantity, \"item2\": quantity, ...}}. "
        "For each food or drink, if the user specifies a quantity or weight, return the estimated weight in grams. If the user does not specify a quantity, return the string 'reference value' for that food item. "
        "Examples: "
        "- '1 bowl of soup' → {\"soup\": 350} "
        "- '2 plates of pasta' → {\"pasta\": 500} "
        "- 'half a cup of rice' → {\"rice\": 75} "
        "- '300g chicken' → {\"chicken\": 300} "
        "- 'I drank a coke' → {\"coke\": 'reference value'} "
        "- 'ate a salad' → {\"salad\": 'reference value'} "
        "If no food items are found, return {\"food_items\": {}}. "
        "IMPORTANT: You MUST end your response with the word TERMINATE on a new line. "
        "Example: {\"food_items\": {\"chicken\": 300, \"soup\": 350, \"pasta\": 500, \"coke\": 'reference value'}} TERMINATE"
    )
    # Create agent
    food_agent = AssistantAgent(
        name="FoodExtractor",
        system_message=system_prompt,
        llm_config=llm_config,
    )
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )
    # Extract food items using LLM
    try:
        chat_result = user_proxy.initiate_chat(food_agent, message=user_input)
        results = chat_result.chat_history[-1]['content'].split('TERMINATE')[0].strip()
        # Remove code block markers if present
        if results.startswith("```"):
            results = results.split("```", 1)[1].strip()
            if results.startswith("json"):
                results = results[4:].strip()
            if results.endswith("```"):
                results = results[:-3].strip()
        # Extract JSON using regex if needed
        match = re.search(r'({.*})', results, re.DOTALL)
        if match:
            results = match.group(1)
        # Try parsing with json, then ast.literal_eval as fallback
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(results)
            except Exception:
                return {
                    'error': 'Could not parse LLM response. Please try again.',
                    'raw_response': results
                }
        food_items = data.get("food_items", {})
        return food_items
    except Exception as e:
        return {
            'error': f'An error occurred: {str(e)}'
        }

def prompt_llm_for_ingredients(dish_name):
    """Prompt the LLM to decompose a dish into basic ingredients, returning estimated weights in grams for each ingredient."""
    llm_config = {"config_list": [{
        "model": "glm-3-turbo",
        "base_url": zhipu_base_url,
        "api_key": zhipu_api_key,
    }]}
    system_prompt = (
        f"Given the dish '{dish_name}', always decompose it into its basic ingredients and their estimated weights in grams, unless it is truly a single-ingredient food. Do NOT just return the dish name or a single ingredient for multi-ingredient dishes. This includes named or branded dishes like 'pepperoni pizza', 'cheeseburger', etc. "
        "Only include ingredients that are basic (e.g., tomatoes, milk, flour, chicken, etc.) and do not decompose further. "
        "Return ONLY a JSON object in the format: {\"ingredients\": {\"ingredient1\": weight_in_grams, ...}}. "
        "For each ingredient, estimate the most likely weight in grams (as a number, no units except grams). Do NOT use units like 'cup', 'head', 'breast', etc. Only use grams. "
        "Examples: "
        "- For 'caesar salad', return: {\"ingredients\": {\"romaine lettuce\": 100, \"croutons\": 30, \"parmesan cheese\": 20, \"caesar dressing\": 40, \"chicken\": 80}} "
        "- For 'spaghetti bolognese', return: {\"ingredients\": {\"spaghetti\": 100, \"ground beef\": 150, \"tomato\": 100, \"onion\": 50, \"garlic\": 5}} "
        "- For 'pepperoni pizza', return: {\"ingredients\": {\"pizza dough\": 120, \"tomato sauce\": 60, \"mozzarella cheese\": 80, \"pepperoni\": 40}} "
        f"If the dish is already a basic ingredient, return: {{\"ingredients\": {{'{dish_name}': 100}}}}. "
        "IMPORTANT: End your response with the word TERMINATE on a new line."
    )
    agent = AssistantAgent(
        name="RecipeDecomposer",
        system_message=system_prompt,
        llm_config=llm_config,
    )
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )
    try:
        chat_result = user_proxy.initiate_chat(agent, message=dish_name)
        results = chat_result.chat_history[-1]['content'].split('TERMINATE')[0].strip()
        # Remove code block markers if present
        if results.startswith("```"):
            results = results.split("```", 1)[1].strip()
            if results.startswith("json"):
                results = results[4:].strip()
            if results.endswith("```"):
                results = results[:-3].strip()
        # Extract JSON using regex if needed
        match = re.search(r'({.*})', results, re.DOTALL)
        if match:
            results = match.group(1)
        # Try parsing with json, then ast.literal_eval as fallback
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(results)
            except Exception:
                print(f"❌ Error decomposing recipe for '{dish_name}': Could not parse LLM response.")
                return {dish_name: 'original amount'}
        return data.get("ingredients", {})
    except Exception as e:
        print(f"❌ Error decomposing recipe for '{dish_name}': {e}")
        return {dish_name: 'original amount'}

def get_basic_ingredient_names(db):
    return list(db[db['Ingredient'] == 1]['Food name'])

def get_composed_dish_names(db):
    return list(db[db['Ingredient'] == 0]['Food name'])

class LLMDBMatcherAgent:
    def __init__(self, food_list, agent_name, model="glm-3-turbo", base_url=None, api_key=None, token_limit=5000, exact_match_only=False):
        self.food_list = food_list
        self.agent_name = agent_name
        self.model = model
        self.base_url = base_url or zhipu_base_url
        self.api_key = api_key or zhipu_api_key
        self.token_limit = token_limit
        self.tokens_used = 0
        self.agent = None
        self.exact_match_only = exact_match_only
        self._init_agent()

    def _init_agent(self):
        food_list_str = '\n'.join([f'{i+1}. {name}' for i, name in enumerate(self.food_list)])
        if self.exact_match_only:
            system_prompt = (
                f"For each ingredient or dish, pick the best match from the following numbered list ONLY if it is a 100% (exact) match. "
                "If there is no exact match, return the string 'none'. "
                "Return ONLY the number of the best match, or 'none', with no extra text or explanation.\n"
                f"{food_list_str}\n"
                "Example: For 'mozzarella', if the best match is 'mozzarella cheese', and it is number 12, return: 12. If there is no exact match, return: none."
            )
        else:
            system_prompt = (
                f"For each ingredient or dish, pick the best match from the following numbered list. "
                "Return ONLY the number of the best match, with no extra text or explanation.\n"
                f"{food_list_str}\n"
                "Example: For 'mozzarella', if the best match is 'mozzarella cheese', and it is number 12, return: 12"
            )
        llm_config = {"config_list": [{
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }]}
        from autogen import AssistantAgent
        self.agent = AssistantAgent(
            name=self.agent_name,
            system_message=system_prompt,
            llm_config=llm_config,
        )
        self.tokens_used = 0

    def match(self, ingredient):
        # Estimate tokens for this prompt
        word_count = len(ingredient.split())
        tokens_this_prompt = int(word_count / 0.75)
        self.tokens_used += tokens_this_prompt
        # If tokens used exceeds limit, refresh agent context
        if self.tokens_used > self.token_limit:
            self._init_agent()
        message = ingredient
        from autogen import UserProxyAgent
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: True,
            code_execution_config={"work_dir": "coding", "use_docker": False},
        )
        try:
            chat_result = user_proxy.initiate_chat(self.agent, message=message)
            result = chat_result.chat_history[-1]['content'].strip().lower()
            # Remove code block markers if present
            if result.startswith("```"):
                result = result.split("```", 1)[1].strip()
                if result.startswith("json"):
                    result = result[4:].strip()
                if result.endswith("```"):
                    result = result[:-3].strip()
            if result == 'none':
                return None
            import re
            match = re.search(r'(\d+)', result)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(self.food_list):
                    return self.food_list[idx]
            return None
        except Exception as e:
            print(f"❌ Error matching ingredient '{ingredient}' to DB: {e}")
            return None 