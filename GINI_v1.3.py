import os
import streamlit as st
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from IPython.display import Markdown, display

# Nutrient Breakdown Prompt Template
NutrientBreakdown_template = """You are an expert nutritionist. Provide a \
detailed nutrient breakdown for the following food item. Include information \
about calories, macronutrients (proteins, fats, carbohydrates), and \
micronutrients (vitamins and minerals) in a concise and easy to understand \
manner. Here is the food item:
{input}"""

# Recipe Suggestions Prompt Template
RecipeSuggestions_template = """You are an expert chef and nutritionist. \
Suggest a list of recipes that can be made with the given ingredients. Each \
recipe should include preparation steps, cooking time, and a nutrient \
breakdown for each serving in a concise and easy to understand manner in a \
Markdown. Make sure the recipes are healthy and balanced. Here are the \
ingredients:
{input}"""

# Meal Plan Prompt Template
MealPlan_template = """You are an expert nutritionist. Create a meal plan \
based on the user's dietary preferences. Each meal should include recipes and \
a detailed nutrient breakdown for each meal in a concise and easy to \
understand manner in a Markdown. The plan should be balanced and cover all the \
necessary nutrients. Here are the user's preferences:
{input}"""

# Healthy Eating Tips Prompt Template
HealthyEatingTips_template = """You are an expert nutritionist. Provide tips \
and advice on healthy eating habits. Include information on portion control, \
balanced diet, hydration, and how to incorporate a variety of nutrients into \
the diet in a concise and easy to understand manner in a Markdown. Make sure \
the tips are practical and easy to follow. Here are the user's preferences or \
concerns:
{input}"""

# Exercise and Nutrition Prompt Template
ExerciseNutrition_template = """You are an expert in both nutrition and \
fitness. Provide a detailed guide on how to align nutrition with an exercise \
regimen. Include pre-workout and post-workout meal suggestions, nutrient \
timing, and hydration tips in a concise and easy to understand manner in a \
Markdown. Here is the user's exercise routine:
{input}"""

# Dietary Restrictions Prompt Template
DietaryRestrictions_template = """You are an expert nutritionist. Provide a \
list of foods that are safe to eat and foods to avoid for someone with the \
following dietary restrictions in a concise and easy to understand manner in a \
Markdown. Also, suggest a few recipes that adhere to these restrictions. \
Here are the dietary restrictions:
{input}"""

# Weight Management Prompt Template
WeightManagement_template = """You are an expert nutritionist. Provide a \
comprehensive guide for weight management. Include tips for both weight loss \
and weight gain, considering healthy eating habits, portion control, and \
exercise recommendations in a concise and easy to understand manner in a \
Markdown. Here is the user's goal:
{input}"""

# Food Allergies Management Prompt Template
FoodAllergies_template = """You are an expert nutritionist. Provide guidance \
on managing food allergies. Include a list of common food allergens, how to \
identify allergic reactions, and safe alternatives for common allergens in a \
concise and easy to understand manner in a Markdown. Here is the user's \
allergy information:
{input}"""

# Digestive Health Prompt Template
DigestiveHealth_template = """You are an expert nutritionist. Provide tips \
and advice on maintaining good digestive health. Include information on foods \
that promote gut health, habits to improve digestion, and how to manage common \
digestive issues in a concise and easy to understand manner in a Markdown. \
Here are the user's concerns:
{input}"""

# Plant-based Diet Prompt Template
PlantBasedDiet_template = """You are an expert nutritionist. Provide guidance \
on following a plant-based diet. Include tips on ensuring adequate protein, \
vitamins, and minerals intake, as well as some balanced meal suggestions in a \
concise and easy to understand manner in a Markdown. Here are the user's \
preferences:
{input}"""

# Prompt information dictionary
prompt_infos = [
    {
        "name": "Nutrient Breakdown",
        "description": "Provides a detailed nutrient breakdown for a specified food item.",
        "prompt_template": NutrientBreakdown_template
    },
    {
        "name": "Recipe Suggestions",
        "description": "Suggests recipes based on given ingredients and provides nutrient breakdown for each.",
        "prompt_template": RecipeSuggestions_template
    },
    {
        "name": "Meal Plan",
        "description": "Creates a meal plan with recipes and nutrient breakdown based on user preferences.",
        "prompt_template": MealPlan_template
    },
    {
        "name": "Healthy Eating Tips",
        "description": "Provides practical tips and advice on healthy eating habits.",
        "prompt_template": HealthyEatingTips_template
    },
    {
        "name": "Exercise and Nutrition",
        "description": "Aligns nutrition with an exercise regimen, providing meal suggestions and nutrient timing.",
        "prompt_template": ExerciseNutrition_template
    },
    {
        "name": "Dietary Restrictions",
        "description": "Provides guidance and recipes for specific dietary restrictions.",
        "prompt_template": DietaryRestrictions_template
    },
    {
        "name": "Weight Management",
        "description": "Offers a comprehensive guide for weight management, including weight loss and gain tips.",
        "prompt_template": WeightManagement_template
    },
    {
        "name": "Food Allergies Management",
        "description": "Provides guidance on managing food allergies and safe alternatives.",
        "prompt_template": FoodAllergies_template
    },
    {
        "name": "Digestive Health",
        "description": "Offers tips and advice on maintaining good digestive health.",
        "prompt_template": DigestiveHealth_template
    },
    {
        "name": "Plant-based Diet",
        "description": "Offers guidance and meal suggestions for following a plant-based diet.",
        "prompt_template": PlantBasedDiet_template
    },
]

# SetUpLLM
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(
    temperature=0.9,
    model=llm_model,
    openai_api_key=os.environ.get("SECRET"))

# SetUpDestinationChainsandRouting
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# SetUpDefaulPromptandChain
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# SetUpMultiPromptRouterTemplate
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a decision input to a \
language model select the model prompt best suited for evaluating the
decision input. You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
REMEMBER: "destination" MUST be one of the candidate prompt
names specified below OR it can be "DEFAULT" if the input is not
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input
if you don't think any modifications are needed.
REMEMBER: Also give me the nutrient breakdown of each meal, if applicable.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""
#SetUpMultiPromptRouterTemplateandRouterPromptandRouterChain
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

#SetUpMultiPromptChain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True)

# Streamlit UI

st.set_page_config(page_title="Gini Nutritionist Chatbot")
st.title("NutriGINI: Generative Informational Nutritional Initiative")
st.write("Ask anything related to nutrition, recipes, meal plans, and more!")
# st.write("Made by Swapnil, Shahrukh, and Aakash!")


decision = st.text_area("Write your query",
"I want to increase my muscle mass, but I do not eat meat, what should be my meal plan for the week?")

nutrient_feature = "/n/n Also give me the nutrient breakdown of each meal"
input_guard= "/n/n NOTE: IF MY PROMPT IS NOT RELATED TO NUTRITION OR HEALTH, PLEASE SAY "'INVALID PROMPT.'""

if st.button("Submit"):
    finished_chain = chain.run(decision + input_guard + nutrient_feature)
    st.markdown(finished_chain, unsafe_allow_html=True)

further_query = st.text_area("Any further query?", "")
if st.button("Submit Further Query"):
    further_query2 = 'FURTHER INSTRUCTIONS:, I would like to clarify that ' + further_query
    further_finished_chain = chain.run(decision + further_query2 + input_guard + nutrient_feature)
    st.markdown(further_finished_chain, unsafe_allow_html=True)

# st.write("<p style='text-align: center;'><i>Made by Swapnil, Shahrukh, and Akash!</i></p>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'><i>Made by <a href='https://www.linkedin.com/in/swapniljha001/'>Swapnil Jha</a>, Shahrukh, and Akash!</i></p>", unsafe_allow_html=True)

