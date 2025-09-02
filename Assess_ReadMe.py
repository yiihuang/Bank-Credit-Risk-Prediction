
import os
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv("/Users/ahmadborzou/F/CF_Bootcamp/.env")

from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(
    temperature=0, model="gpt-4o-mini", 
    api_key=os.getenv("openai_api_key"))


def assess_ReadMe_with_checklist(ReadMe_content):
    # Reorganized checklist for improved AI model performance:
    # - Group related criteria under clear, concise headings
    # - Use bullet points for clarity
    # - Reduce redundancy and ambiguity
    # - Place bonus/formatting at the end

    checklist = f"""
## README Assessment Rubric

### 1. Project Overview (0–3 pts)
- Clearly states project purpose, domain, and context
- Mentions dataset/problem being solved
- Specifies intended audience

### 2. File & Folder Structure (0–3 pts)
- Lists all major files/folders
- Describes each file/folder's purpose
- Structure matches actual repository

### 3. Setup & Installation (0–4 pts)
- Specifies environment requirements (Python version, OS, etc.)
- Lists dependencies and installation steps
- Explains virtual environment setup (if any)
- Details API keys, data downloads, or config setup

### 4. Usage & Execution Guide (0–6 pts)
- Explains entry point/main script and how to run it
- Describes each file's role in simple terms
- Shows execution order (e.g., which scripts to run first)
- Provides example commands for running scripts
- Explains expected outputs/logs
- Includes screenshots/diagrams (optional, bonus)

### 5. Results & Interpretation (0–3 pts)
- Summarizes model performance or outcomes
- Provides metrics, plots, or screenshots
- Reflects on results or suggests improvements

### 6. Formatting & Clarity (Bonus, 0–3 pts)
- Clean, readable Markdown (headings, lists, code blocks)
- Table of contents or navigation links
- Consistent, error-free English

### Scoring
- Base Total: /19
- With Bonus: /22

Assess the following README content:
---
{ReadMe_content}
---
Return your answer as a list of objects, one per section, plus a final summary object.
"""

    schema = {
        "title": "ReadMeAssessment",
        "description": "Assessment of a README file based on a reorganized checklist, with scores, notes, and subtotals for each section.",
        "type": "object",
        "properties": {
            "assessment": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string"},
                        "criteria": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "score": {"type": "integer"},
                                    "notes": {"type": "string"}
                                },
                                "required": ["description", "score", "notes"]
                            }
                        },
                        "subtotal": {"type": "integer"},
                        "max_points": {"type": "integer"}
                    },
                    "required": [
                        "section", "criteria", "subtotal", "max_points"
                    ]
                },
                "description": "List of assessment results for each checklist section, including criteria, scores, notes, and subtotals."
            },
            "bonus": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "score": {"type": "integer"},
                                "notes": {"type": "string"}
                            },
                            "required": ["description", "score", "notes"]
                        }
                    },
                    "subtotal": {"type": "integer"},
                    "max_points": {"type": "integer"}
                },
                "required": ["criteria", "subtotal", "max_points"]
            },
            "total_score": {"type": "integer"},
            "total_max_points": {"type": "integer"},
            "total_with_bonus": {"type": "integer"}
        },
        "required": ["assessment", "bonus", "total_score", "total_max_points", "total_with_bonus"]
    }


    llm_struc = llm.with_structured_output(schema)
    result = llm_struc.invoke(checklist)
    return result




with open("ReadMe.md", "r", encoding="utf-8") as f:
    ReadMe_content = f.read()

assessment_result = assess_ReadMe_with_checklist(ReadMe_content)
print(assessment_result)


with open("ReadMe_assessment.json", "w", encoding="utf-8") as json_file:
    json.dump(assessment_result, json_file, indent=4, ensure_ascii=False)

    