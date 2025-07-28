# medinfo_intent_app.py
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load intent schema from Excel
df = pd.read_excel("medinfo_agent_list_of_intents.xlsx", engine="openpyxl", header=None)

# Define intents
medical_intents = []
for i in range(13, 21):
    row = df.iloc[i]
    intent = {
        "intent_id": f"medical_{row[1]}",
        "category": "Medical",
        "subtype": row[2],
        "definition": row[3],
        "keywords": row[4].split(",") if pd.notna(row[4]) else [],
        "response": None
    }
    medical_intents.append(intent)

nonmedical_intents = []
for i in range(26, 39):
    row = df.iloc[i]
    intent = {
        "intent_id": f"nonmedical_{row[1]}",
        "category": "Non-Medical",
        "subtype": row[2],
        "definition": row[5],
        "keywords": row[6].split(",") if pd.notna(row[6]) else [],
        "response": row[7] if pd.notna(row[7]) else ""
    }
    nonmedical_intents.append(intent)

all_intents = medical_intents + nonmedical_intents

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
intent_embeddings = [(intent, model.encode(f"{intent['subtype']}. {intent['definition']}. {' '.join(intent['keywords'])}", convert_to_tensor=True)) for intent in all_intents]

# Streamlit UI
st.title("Ibrance Patient Inquiry Classifier")
user_query = st.text_input("Enter patient inquiry below:")

if user_query:
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    best_intent, best_score = None, -1
    for intent, emb in intent_embeddings:
        score = util.pytorch_cos_sim(query_embedding, emb).item()
        if score > best_score:
            best_score, best_intent = score, intent

    st.subheader("Classification Result")
    st.write(f"**Category:** {best_intent['category']}")
    st.write(f"**Subtype:** {best_intent['subtype']}")
    st.write(f"**Confidence:** {best_score:.2f}")

    if best_intent['category'] == "Non-Medical":
        st.write(f"**Response:** {best_intent['response']}")
    else:
        # 7/27 fix unicode runtime err
        #Use LLM to generate response grounded in Ibrance PI
        #with open("USPI_Ibrance_palbociclib_capsules.md") as f1, open("USPI_Ibrance_palbociclib_TABLETS.md") as f2:
        #    usp_info = f1.read() + "\n" + f2.read()

        with open("USPI_Ibrance_palbociclib_capsules.md", encoding="utf-8") as f1, \
          open("USPI_Ibrance_palbociclib_TABLETS.md", encoding="utf-8") as f2:
          usp_info = f1.read() + "\n" + f2.read()

        prompt = f"Using only the information below, draft a medical response to the following patient question: \"{user_query}\"\n\n{usp_info}"
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical assistant responding strictly based on drug label content."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        st.write("**Response (Medical Draft):**")
        st.markdown(answer)

    # Feedback option
    if st.checkbox("Flag this classification as incorrect"):
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "predicted_intent_id": best_intent["intent_id"],
            "category": best_intent["category"],
            "subtype": best_intent["subtype"],
            "confidence": round(best_score, 2),
            "flagged": True
        }
        feedback_log = []
        if os.path.exists("feedback_log.json"):
            with open("feedback_log.json") as f:
                feedback_log = json.load(f)
        feedback_log.append(feedback)
        with open("feedback_log.json", "w") as f:
            json.dump(feedback_log, f, indent=2)
        st.success("Feedback submitted successfully.")

    # Save to Excel for audit
    log_df = pd.DataFrame([{
        "Timestamp": datetime.now(),
        "Query": user_query,
        "Intent": best_intent["subtype"],
        "Category": best_intent["category"],
        "Confidence": best_score,
        "Response": best_intent["response"] if best_intent['category'] == "Non-Medical" else answer,
        "Flagged": True if st.session_state.get("feedback") else False
    }])
    with pd.ExcelWriter("intent_logs.xlsx", engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        log_df.to_excel(writer, sheet_name="intents", index=False)

# Feedback dashboard
if st.sidebar.checkbox("Show Feedback Review Dashboard"):
    if os.path.exists("feedback_log.json"):
        with open("feedback_log.json") as f:
            feedback_data = json.load(f)
        st.sidebar.write(pd.DataFrame(feedback_data))
    else:
        st.sidebar.write("No feedback yet.")
