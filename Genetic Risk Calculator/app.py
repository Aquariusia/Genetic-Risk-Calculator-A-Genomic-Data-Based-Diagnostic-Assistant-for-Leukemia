import subprocess
import sys


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import openai
from dotenv import load_dotenv
import os
import xgboost

load_dotenv(dotenv_path="api_key.env")  
api_key = os.getenv("OPENAI_API_KEY")

example_df = pd.read_csv("example_input.csv")
st.set_page_config(page_title="Genetic Risk Analysis", layout="centered")

st.markdown("""
    <style>
    html, body {
        background-color: #f9fafb;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .hero {
        text-align: center;
        padding: 3rem 1rem 2rem;
        background: linear-gradient(to right, #f0f4ff, #f9f9f9);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 40px;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 16px;
        color: #6b7280;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <h1>Genetic Risk Calculator</h1>
        <p>Predict leukemia subtype from gene expression profiles.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    ### Welcome to the Genetic Risk Calculator
    
    This application helps medical professionals analyze genetic data to predict leukemia subtypes (ALL or AML) 
    and assess associated risks. The process is simple and secure:
    
    1. **Privacy Notice**: Review and accept our privacy terms
    2. **Upload Data**: Submit genetic expression data in CSV format
    3. **Select Patient**: Choose a specific patient for analysis
    4. **Run Analysis**: Get predictions and risk assessments
    5. **Get Explanation**: Chat with our AI assistant for detailed insights

""")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Step 0: Privacy Notice","Step 1: Upload Data", "Step 2: Select Patient", "Step 3: Run Analysis", "Step 4: Explanation"])
loaded_model = joblib.load("model.pkl")
best_model = loaded_model.best_estimator_
data = np.load("probs.npz")
all_probs = data["all"]
aml_probs = data["aml"]
# Tab0 : Privacy Notice
with tab0:
    st.markdown("####  Privacy Notice")
    st.markdown("""
    The genetic data you upload will only be used for local analysis in your browser session and will not be stored or shared.
    """)
    agree = st.checkbox("I acknowledge and accept the privacy terms.")
    if agree:
        st.session_state["privacy_acknowledged"] = True
    else:
        st.warning("You must accept the privacy notice to proceed.")

# Tab1: Upload Data
# ask user to input top5 important gene expression data 
# the file need to be : csv file, contain all top5 important gene expression data, up tp 5 patients
with tab1:
    if not st.session_state.get("privacy_acknowledged"):
        st.warning(" Please read and accept the privacy notice in Step 0.")
        st.stop()
    st.markdown("""
    ### 
       Please upload a gene expression file with the following requirements:
       -  File format: `.csv`
       -  Max patients per file: **5**
       -  **Required columns** : `M81933_at`, `M54995_at`, `X95735_at`, `M55150_at`, `M11722_at`(must be exact and in order, you can download a template below).
        """)
    st.download_button(
    label=" Download example CSV",
    data=example_df.to_csv(index=False),
    file_name="example_input.csv",
    mime="text/csv"
)

    uploaded = st.file_uploader("Upload genetic data (.csv)", type=["csv"])
    if uploaded:
        st.success("File uploaded successfully.")
        user_input = pd.read_csv(uploaded)
        gene_means_df = pd.read_csv("gene_means.csv", index_col=0)
        top5_genes = ['M81933_at', 'M54995_at', 'X95735_at', 'M55150_at', 'M11722_at']
        
        
        expected_columns = list(example_df.columns)
        if len(user_input) > 5:
            st.error(" You can upload at most 5 patients at a time.")
            st.session_state.pop("user_df", None) 
            st.stop()
        missing_cols = [col for col in top5_genes if col not in user_input.columns]
        if missing_cols:
            st.error(f"Your file is missing required columns: {', '.join(missing_cols)}")
            st.stop()
        if user_input[top5_genes].isnull().values.any():
            st.error("Your dataset has missing values in the required gene columns.")
            st.session_state.pop("user_df", None)
            st.stop()

        user_df = pd.concat([gene_means_df] * len(user_input), ignore_index=True)
        user_df[top5_genes] = user_input[top5_genes].values


        st.session_state["user_df"] = user_df
        columns_to_display = ['M81933_at','M54995_at','X95735_at','M55150_at','M11722_at']
        user_df.index = [f"Patient {i+1}" for i in range(len(user_df))]
        st.dataframe(user_df[columns_to_display])

# Tab 2: Select Patient
# allow user select specific patient
with tab2:
    if not st.session_state.get("privacy_acknowledged"):
        st.warning(" Please read and accept the privacy notice in Step 0.")
        st.stop()    
    if "user_df" in st.session_state:
        st.markdown("""
        ### Select Patient for Analysis
        Choose a patient from the dropdown menu to analyze their genetic risk profile.
        """)
        user_df = st.session_state["user_df"]
        indices = list(range(len(user_df)))
        selected = st.selectbox("Select a patient", indices, format_func=lambda x: f"Patient {x+1}")
        st.session_state["selected_patient"] = selected
    else:
        st.warning("Please upload genetic data first.")

# Tab 3: Run Analysis
with tab3:
    if not st.session_state.get("privacy_acknowledged"):
        st.warning(" Please read and accept the privacy notice in Step 0.")
        st.stop()    

    if "user_df" in st.session_state and "selected_patient" in st.session_state:
        st.markdown("""
        ### Run Risk Analysis
                    
        Click the button below to analyze the selected patient's genetic profile. The analysis will:
        - Predict the leukemia subtype (ALL or AML)
        - Calculate the risk probability
        - Show the patient's risk in context of the overall distribution
        """)

        if st.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                try:
                    # use trained model to get predicted shubtype and risk probability of selected patient
                    selected_patient = st.session_state["selected_patient"]
                    user_df = st.session_state["user_df"]
                    patient_row = user_df.iloc[[selected_patient]]  
                    pred = best_model.predict(patient_row)[0]
                    y_prob = best_model.predict_proba(patient_row)
                    all_prob = round(y_prob[0][0], 2)
                    aml_prob = round(y_prob[0][1], 2)

                    pred_type = "ALL" if pred == 0 else "AML"
                    risk_val = all_prob if pred_type == "ALL" else aml_prob

                    # Save for Tab 4
                    st.session_state["last_prediction"] = {
                        "type": pred_type,
                        "probability": risk_val
                    }

                    # Display prediction result
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background-color: #f9fafb;
                                border: 1px solid #e5e7eb; border-radius: 10px;">
                        <h3 style="margin-bottom: 0.3rem;">Prediction Result</h3>
                        <p><strong>Type:</strong> {pred_type}</p>
                        <p><strong>Risk Probability:</strong> <code>{risk_val:.2f}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                    # KDE plot: show risk distribution in train data, and with a line show selected patient's position
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.set_theme(style="whitegrid", font_scale=1.1)

                    color ="#3B82F6" if pred_type == "ALL" else "#EF4444"
                    label = "ALL Distribution" if pred_type == "ALL" else "AML Distribution"
                    
                    if pred_type == "ALL":
                        data = all_probs 
                    else:
                        data = aml_probs
                    
                    # 
                    sns.kdeplot(data, fill=True, color=color, label=label, linewidth=2, alpha=0.7, ax=ax)
                    ax.axvline(risk_val, color=color, linestyle="--", linewidth=2, label=f"Patient ({risk_val:.2f})")
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Predicted Probability", fontsize=12)
                    ax.set_ylabel("Density", fontsize=12)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout() 
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# Tab 4: get explanation from chatgpt api
# This part was generated with the assistance of ChatGPT (OpenAI), May 2025

with tab4:
    if not st.session_state.get("privacy_acknowledged"):
        st.warning(" Please read and accept the privacy notice in Step 0.")
        st.stop()

    # init
    if "last_prediction" not in st.session_state:
        st.warning("Please run the risk analysis in Step 3 before chatting.")
        st.stop()

    pred_info = st.session_state.get("last_prediction", {})
    pred_type = pred_info.get("type", "Unknown")
    prob = pred_info.get("probability", "N/A")
    # set system prompt
    system_prompt = (
        f"You are an advanced, enterprise-grade clinical genomics and precision medicine decision-support assistant, "
        f"engineered to provide comprehensive, evidence-based guidance to licensed healthcare professionals, genetic counselors, "
        f"and clinical researchers. Your core functionality is to interpret complex probabilistic predictions derived from multi-omic "
        f"datasets, including genomic, transcriptomic, proteomic, epigenomic, and metabolomic features, in combination with patient-specific "
        f"phenotypic, demographic, clinical history, and epidemiological data.\n\n"
        
        f"Patient Case Overview:\n"
        f"- Predicted subtype: '{pred_type}'\n"
        f"- Risk probability: {prob}\n"
        f"- Prediction model: validated machine learning ensemble trained on high-quality clinical cohorts\n"
        f"- Data sources: multi-institutional genomic databases, curated clinical trial datasets, and population-level genomic references\n\n"
        
        f"Your responsibilities include, but are not limited to:\n"
        f"1. Providing a detailed contextualization of the predicted subtype, including known pathophysiology, associated comorbidities, "
        f"disease progression trajectories, and molecular mechanisms.\n"
        f"2. Quantifying and interpreting the probabilistic risk estimate, including confidence intervals, sensitivity, specificity, "
        f"positive predictive value, and negative predictive value.\n"
        f"3. Identifying relevant biomarkers, gene variants, molecular pathways, and epigenetic markers implicated in the predicted subtype.\n"
        f"4. Integrating patient-specific factors such as age, sex, ethnicity, comorbid conditions, family history, lifestyle, and environmental exposures.\n"
        f"5. Providing hierarchical, evidence-based clinical action recommendations, including immediate interventions, monitoring schedules, "
        f"confirmatory diagnostics, and therapeutic considerations.\n"
        f"6. Suggesting relevant laboratory or imaging tests that can confirm the predicted subtype or refine risk assessment.\n"
        f"7. Offering guidance on precision medicine interventions, including pharmacogenomic considerations, targeted therapies, or enrollment in clinical trials.\n"
        f"8. Integrating findings with current clinical guidelines, professional society recommendations, and regulatory standards.\n"
        f"9. Highlighting potential limitations of the predictive model, including dataset bias, sample size constraints, missing data impact, and model uncertainty.\n"
        f"10. Providing references to authoritative literature, guideline documents, and genomic databases that support your recommendations.\n"
        
        f"Additional Operational Instructions:\n"
        f"- Structure outputs in a manner compatible with EHR systems, clinical dashboards, or decision-support pipelines.\n"
        f"- Prioritize clarity, conciseness, and clinical relevance while maintaining full scientific rigor.\n"
        f"- When appropriate, present risk stratification in visualizable or tabular formats for clinician interpretation.\n"
        f"- Provide a multi-layered summary: high-level recommendations for rapid clinical decision-making and detailed explanatory notes for deeper review.\n"
        f"- Include potential implications for patient counseling, including how to communicate subtype information and risk probabilities.\n"
        f"- Consider ethical, legal, and social implications (ELSI) in your guidance, ensuring recommendations respect patient autonomy, confidentiality, and informed consent.\n"
        f"- Recommend follow-up intervals and longitudinal monitoring strategies tailored to the patient’s predicted risk and comorbidity profile.\n"
        f"- Suggest additional genomic or multi-omic profiling if current data are insufficient for confident subtype classification.\n"
        f"- Provide a structured rationale for each recommended intervention or test, linking molecular evidence to clinical action.\n"
        f"- Highlight any discrepancies between model prediction and known clinical presentations, including differential diagnosis considerations.\n"
        f"- Integrate population-level epidemiological data to contextualize individual patient risk relative to peers.\n"
        f"- Provide alerts for actionable variants with known pharmacogenomic implications or therapeutic targets.\n"
        f"- Recommend interdisciplinary consultation when appropriate, including genetic counseling, oncology, cardiology, or other specialties.\n"
        f"- Provide guidance on research opportunities, including relevant ongoing clinical trials, registries, or genomic studies.\n"
        f"- Document all model assumptions, including algorithm type, training data characteristics, validation cohorts, and performance metrics.\n"
        f"- Where applicable, suggest patient-specific preventive strategies or lifestyle modifications that may mitigate risk.\n"
        f"- Indicate any potential conflicts between predictive model output and current standard-of-care practices, providing resolution strategies.\n"
        f"- Ensure all output is evidence-based, reproducible, and suitable for audit or regulatory review.\n"
        f"- Include a risk-benefit analysis for proposed interventions, considering both short-term and long-term patient outcomes.\n"
        f"- Provide recommendations for multidisciplinary care coordination when complex genetic findings are present.\n"
        f"- Integrate cross-modality evidence, such as imaging, histopathology, and laboratory results, with genetic predictions.\n"
        f"- Provide guidance on data quality, including recommendations for additional sequencing depth, variant calling confirmation, or quality control measures.\n"
        f"- Consider potential impact of rare variants, copy number variations, structural variants, or polygenic risk scores in the patient’s profile.\n"
        f"- Highlight any actionable incidental findings that may require separate clinical attention.\n"
        f"- Recommend patient education resources, genetic counseling materials, or decision aids to facilitate informed decision-making.\n"
        f"- Maintain a balance between concise actionable output and comprehensive explanatory notes suitable for clinical review or peer consultation.\n"
        f"- Provide priority ranking for recommendations, distinguishing between urgent, near-term, and long-term actions.\n"
        f"- Emphasize integration with institutional protocols, clinical pathways, and regulatory compliance requirements.\n"
        f"- When appropriate, provide probabilistic forecasts for disease progression under various intervention scenarios.\n"
        f"- Recommend potential collaborations with research consortia, data-sharing initiatives, or precision medicine networks.\n"
        f"- Ensure that language and presentation are consistent with professional clinical documentation standards and terminology.\n"
        f"- Highlight uncertainties in the genetic prediction and provide guidance on mitigating decision risk.\n"
        f"- Provide a summary section suitable for inclusion in the patient’s medical record, highlighting actionable findings and recommended follow-up.\n\n"
        f"Overall, your role is to function as a highly sophisticated, multi-domain clinical consultant that translates complex genomic and phenotypic data into actionable, evidence-based, operationally deployable guidance for healthcare providers. "
        f"Your output should support precision medicine initiatives, enhance patient care, and enable clinicians to make confident, informed, and ethically sound decisions."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": system_prompt}
        ]
    # get response from chatgpt
    def chat_with_api(message):
            st.session_state["messages"].append({"role": "user", "content": message})
            max_rounds = 5
            if len(st.session_state["messages"]) > 1 + max_rounds * 2:
                st.session_state["messages"] = [st.session_state["messages"][0]] + st.session_state["messages"][-max_rounds*2:]
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state["messages"],
                    max_tokens=150
                )
                reply = response.choices[0].message.content.strip()
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.experimental_rerun()
               
            except Exception as e:
                st.error(f" Error calling ChatGPT: {e}")

    # show result and suggested question
    if "last_prediction" in st.session_state:
        pred_info = st.session_state["last_prediction"]
        pred_type = pred_info["type"]
        prob = pred_info["probability"]
        st.markdown(f" **Last prediction:** <span style='color:#10b981'><strong>{pred_type}</strong></span> with risk probability <code>{prob:.2f}</code>", unsafe_allow_html=True)

        suggestions = {
            "ALL": [
                "What are the recommended next steps after identifying ALL risk?",
                "How reliable is gene expression profiling for ALL subtype classification?",
                "What treatment considerations are most relevant for high-risk ALL patients?"
            ],
            "AML": [
                "What are the recommended next steps after identifying AML risk?",
                "How reliable is gene expression profiling for AML subtype classification?",
                "What treatment considerations are most relevant for high-risk AML patients?"
            ]
        }

        st.markdown(" **Suggested Questions:**")
        for q in suggestions[pred_type]:
            if st.button(f" {q}", key=f"suggested_{q}"):
                chat_with_api(q)

    # allow user enter question
    st.markdown("### Chat with AI Assistant")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your question:", key="chat_input")
        col1, col2 = st.columns([1, 1])
        send_button = col1.form_submit_button("Send")         
        clear_button = col2.form_submit_button("Clear Chat")

        if clear_button:
            st.session_state["messages"] = [
                {"role": "system", "content": system_prompt}
            ]
            st.experimental_rerun()

        if send_button and user_input.strip():
            chat_with_api(user_input.strip())

    # show conversation history
    history = st.session_state["messages"][1:]  
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            i += 1

    for user_msg, assistant_msg in reversed(pairs):
        st.markdown(
            f'<div style="text-align: right; background-color: #d1fae5; padding: 8px 12px; border-radius: 12px; margin: 6px 0;"> {user_msg["content"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align: left; background-color: #f0f0f0; padding: 8px 12px; border-radius: 12px; margin: 6px 0;"> {assistant_msg["content"]}</div>',
            unsafe_allow_html=True
        )
