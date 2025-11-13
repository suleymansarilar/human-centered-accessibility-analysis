# GenAI-Guided Accessibility Dashboard

This Streamlit prototype visualizes basic accessibility metrics for six sample neighborhoods and can generate simple narratives for each one.

## Demo
- Video walkthrough: [Watch the demo](https://drive.google.com/file/d/12OXw9Lt5ONIdFrYVRzDCEY4SjRMbzup8/view?usp=drive_link)

## Requirements
`
pip install -r requirements.txt
`

## Run Locally
`
python -m streamlit run app.py
`
- If the browser does not open automatically, visit http://127.0.0.1:8501.

## Features
- Bar chart of accessibility scores per neighborhood
- Scatter chart comparing walking distance and transit nodes
- Rule-based narrative summary with optional LLM output (requires openai package and OPENAI_API_KEY)

## Optional LLM Integration
`
pip install openai
setx OPENAI_API_KEY "YOUR_KEY"
`
Then enable the checkbox inside the app sidebar to switch to the LLM-generated narrative.
