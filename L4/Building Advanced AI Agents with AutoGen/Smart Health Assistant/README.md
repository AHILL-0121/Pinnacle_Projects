# 🏥 Smart Health Assistant

A multi-agent health assessment system using **AutoGen** for personalized BMI calculation, health recommendations, and meal planning.

## 🎯 Overview

This project implements a sequential multi-agent workflow for comprehensive health assessment:
- Collects user health information (height, weight, age, activity level)
- Calculates BMI using specialized function calling
- Provides health category classification and recommendations
- Generates personalized meal plans based on health goals
- Coordinates workflow through agent handoffs

## 🏗️ Architecture

### Sequential Agent Workflow

```
User Proxy Agent
    ↓ (Collect: height, weight, age, activity)
BMI Agent
    ↓ (Calculate BMI, classify health category)
Meal Planning Agent
    ↓ (Generate personalized meal plan)
Summary Agent
    ↓ (Consolidate recommendations)
    ↓
HEALTH_PLAN_COMPLETE ✅
```

### Agent Roles

| Agent | Responsibility | Tools Used |
|-------|----------------|------------|
| **User Proxy** | Collects health information from user | Human input |
| **BMI Agent** | Calculates BMI using function call, classifies health category | `calculate_bmi()` |
| **Meal Planning Agent** | Creates personalized meal plans for health goals | LLM-based generation |
| **Summary Agent** | Consolidates all recommendations into final report | Report generation |

### BMI Categories

| BMI Range | Classification | Health Status |
|-----------|----------------|---------------|
| < 18.5 | **Underweight** | ⚠️ Risk of nutritional deficiency |
| 18.5 - 24.9 | **Normal** | ✅ Healthy weight |
| 25.0 - 29.9 | **Overweight** | ⚠️ Risk of health issues |
| ≥ 30.0 | **Obese** | 🚨 High health risk |

## 🔑 Key Features

- **Function Calling**: BMI calculation using AutoGen's tool registration
- **Health Classification**: Automatic categorization based on BMI
- **Personalized Recommendations**: Tailored advice for each BMI category
- **Meal Planning**: Customized meal plans for weight goals (loss/gain/maintenance)
- **Activity-Aware**: Considers user's activity level for calorie recommendations
- **Termination Signal**: Clear workflow completion with `HEALTH_PLAN_COMPLETE`

## 🛠️ Technical Specifications

| Component | Technology |
|-----------|-----------|
| **Framework** | AutoGen 0.11+ |
| **LLM Provider** | Groq (llama-3.1-70b-versatile) |
| **Agent Pattern** | Sequential Multi-Agent with Function Calling |
| **Function Tools** | `calculate_bmi()` registered with BMI Agent |
| **Termination Strategy** | Keyword-based (`HEALTH_PLAN_COMPLETE`) |

## 📋 Prerequisites

- Python 3.10+
- Groq API Key ([Get it here](https://console.groq.com))
- Basic understanding of health metrics

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install autogen groq
```

### 2. Set Up API Key

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"
```

Or use Google Colab secrets:

```python
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

### 3. Run the Notebook

Open `smart_health_assistant.ipynb` in Jupyter or Google Colab and run all cells.

### 4. Provide Health Information

When prompted, enter your health details:

```
Please provide your health information:

Weight (kg): 85
Height (m): 1.75
Age (years): 30
Activity Level (sedentary/moderate/active): moderate
Health Goal (weight_loss/weight_gain/maintenance): weight_loss
```

## 📊 Example Workflow

### Input

```
Weight: 85 kg
Height: 1.75 m
Age: 30 years
Activity Level: Moderate
Health Goal: Weight Loss
```

### BMI Calculation

```
🔍 BMI Agent Calculating...

Your BMI: 27.8
Category: OVERWEIGHT ⚠️

Health Recommendations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Recommended Weight Range: 67-75 kg
• Weight to Lose: 10-18 kg
• Target BMI: 22.0 (middle of healthy range)

Lifestyle Recommendations:
1. Create a calorie deficit of 500-750 kcal/day
2. Aim for 0.5-1 kg weight loss per week
3. Mix cardio (150 min/week) with strength training (2x/week)
4. Stay hydrated (3-4 liters water daily)
5. Get 7-8 hours of quality sleep
```

### Meal Plan

```
🥗 Personalized Meal Plan (Weight Loss)

Daily Calorie Target: 1,800 kcal
Macros: Protein 30% | Carbs 40% | Fats 30%

━━━━ BREAKFAST (450 kcal) ━━━━
• 3 egg white omelette with spinach & tomatoes
• 2 slices whole wheat toast
• 1 cup green tea
• 1 small apple

━━━━ MID-MORNING SNACK (150 kcal) ━━━━
• 1 cup Greek yogurt
• Handful of almonds (10-12)

━━━━ LUNCH (550 kcal) ━━━━
• Grilled chicken breast (150g)
• Brown rice (1 cup cooked)
• Mixed vegetable salad
• Lemon & olive oil dressing

━━━━ EVENING SNACK (200 kcal) ━━━━
• Protein smoothie (whey + banana + almond milk)

━━━━ DINNER (450 kcal) ━━━━
• Grilled fish (salmon/tilapia, 150g)
• Quinoa (½ cup cooked)
• Roasted vegetables (broccoli, carrots)
• Side of cucumber raita

Tips:
✓ Eat every 3-4 hours to maintain metabolism
✓ Drink water before meals to control portions
✓ Avoid processed foods and sugary drinks
✓ Track calories using apps (MyFitnessPal, HealthifyMe)
```

### Summary

```
📋 COMPREHENSIVE HEALTH PLAN

BMI Status: Overweight (27.8)
Goal: Weight Loss (10-18 kg over 10-18 weeks)

Action Plan:
1. Follow the meal plan strictly for 4 weeks
2. Exercise 5 days/week (cardio + strength training)
3. Track weight weekly (same time, same conditions)
4. Adjust calories if weight loss plateaus
5. Consult a doctor before starting any new program

⚠️ Disclaimer: This is AI-generated guidance. Consult a 
healthcare professional for personalized medical advice.

HEALTH_PLAN_COMPLETE ✅
```

## 🔧 Configuration

### LLM Configuration

```python
llm_config = {
    "model": "llama-3.1-70b-versatile",
    "api_key": os.environ.get("GROQ_API_KEY"),
    "temperature": 0.3,  # Lower for factual health advice
    "max_tokens": 2000,
}
```

### BMI Calculation Function

```python
def calculate_bmi(weight: float, height: float) -> dict:
    """
    Calculates BMI and returns classification with recommendations.
    
    Args:
        weight (float): Weight in kilograms
        height (float): Height in meters
    
    Returns:
        dict: BMI value, category, and health recommendations
    """
    bmi = weight / (height ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
        advice = "Focus on nutrient-dense foods to gain healthy weight."
    elif 18.5 <= bmi < 25:
        category = "Normal"
        advice = "Maintain current lifestyle with balanced diet and exercise."
    elif 25 <= bmi < 30:
        category = "Overweight"
        advice = "Create calorie deficit through diet and exercise."
    else:
        category = "Obese"
        advice = "Consult healthcare professional for comprehensive weight management."
    
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "advice": advice
    }
```

### Activity Level Multipliers

| Activity Level | Daily Calorie Multiplier |
|----------------|--------------------------|
| **Sedentary** | BMR × 1.2 (little/no exercise) |
| **Moderate** | BMR × 1.55 (exercise 3-5 days/week) |
| **Active** | BMR × 1.725 (exercise 6-7 days/week) |

### Termination Condition

```python
is_termination_msg = lambda x: "HEALTH_PLAN_COMPLETE" in x.get("content", "")
```

## 🎓 Health Goals Supported

### 1. Weight Loss
- Calorie deficit meal plans
- Cardio + strength training recommendations
- Portion control strategies

### 2. Weight Gain
- Calorie surplus meal plans
- Protein-rich food suggestions
- Strength training focus

### 3. Maintenance
- Balanced macro distribution
- Sustainable lifestyle habits
- Fitness maintenance routines

## 🎓 Learning Outcomes

This project demonstrates:

1. **Function Calling**: Registering and using custom tools with AutoGen
2. **Sequential Coordination**: Agents passing context through handoffs
3. **Health Domain Modeling**: Implementing BMI classification and recommendations
4. **Personalization**: Tailoring outputs based on user inputs
5. **Termination Signals**: Using keywords for workflow completion
6. **Multi-Step Reasoning**: Breaking complex task into specialized sub-tasks

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'autogen'`
- **Solution**: Install AutoGen: `pip install autogen`

**Issue**: `Invalid API Key`
- **Solution**: Verify your Groq API key is correctly set in environment variables

**Issue**: `BMI function not called`
- **Solution**: Ensure function is registered correctly with `register_for_execution()` and `register_for_llm()`

**Issue**: `Meal plan not personalized`
- **Solution**: Check that user data is passed correctly through agent chain

**Issue**: `Workflow not terminating`
- **Solution**: Verify `HEALTH_PLAN_COMPLETE` is included in final agent's message

## 📚 Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AutoGen Function Calling](https://microsoft.github.io/autogen/docs/tutorial/tool-use)
- [BMI Calculation Guide](https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/body-mass-index)
- [Nutrition Guidelines](https://www.nutrition.gov/)

## ⚠️ Disclaimer

This tool provides **educational health insights** and should NOT be considered professional medical advice. Always consult:
- Healthcare professional for medical conditions
- Registered dietitian for meal planning
- Fitness trainer for exercise programs
- Doctor before starting any weight loss/gain program

## 📄 License

This project is part of the Pinnacle Projects portfolio.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more health metrics (body fat %, muscle mass)
- Implement exercise routine generator
- Add regional cuisine support (Indian, Mediterranean, etc.)
- Create progress tracking functionality
- Add water intake and sleep recommendations

---

**Part of the Pinnacle Projects - L4: Building Advanced AI Agents with AutoGen**
