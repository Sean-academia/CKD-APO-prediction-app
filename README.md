# CKD Pregnancy APO Prediction Web App

This folder contains the clean deployment package for the Streamlit prediction tool.

## Included Files

- `app.py`: Streamlit web app.
- `saved_models/RF.pkl`: final RandomForestClassifier model.
- `saved_models/scaler.pkl`: fitted scaler used before prediction.
- `saved_models/feature_names.pkl`: final 11 model input features.
- `assets/ckd_pregnancy_hero.png`: homepage image.
- `llm_interpretation_cache/`: optional cache for local LLM-generated interpretations.

No raw patient data, imputed datasets, training sets, validation sets, or test sets are included in this deployment package.

The final model uses the following predictors in this fixed order:

`ppHTN`, `ppCHD`, `t1Hb`, `t1TSH`, `t1TG`, `t224hUP`, `t2UA`, `t2WBC`, `t2FT4`, `t2FPG`, and `t2MAP`.

## Run The App

```bash
cd deploy_app
streamlit run app.py
```

## Manual Server Deployment

1. Copy the complete `deploy_app` folder to the target server.
2. Create and activate a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the web app:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

5. Visit the app from a browser:

```text
http://<server-ip>:8501
```

If the server is placed behind a hospital intranet or reverse proxy, expose only the approved internal address.

## Docker Deployment

```bash
cd deploy_app
docker build -t ckd-apo-prediction-app .
docker run -p 8501:8501 ckd-apo-prediction-app
```

Then open:

```text
http://127.0.0.1:8501
```

## Local LLM Interpretation

The prediction model and SHAP visualization can run without a local LLM.

To enable LLM-assisted individualized interpretation:

1. Open LM Studio.
2. Load `qwen2.5-7b-instruct`.
3. Start the Local Server at `http://127.0.0.1:1234`.
4. The app will call the OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1`.

The LLM interpretation module uses a structured prompt template and local caching. For identical patient inputs, model output, SHAP contributions, language, and prompt version, the app returns the same cached interpretation text.

For server deployment, the local LLM server must run on the same machine as the Streamlit app, or the `llm_base_url` in `app.py` must be changed to the approved internal LLM server address.

## Notes

- The LLM interpretation is optional and is not part of the machine-learning prediction model.
- The LLM should only translate model-predicted risk and SHAP-derived feature contributions into natural language.
- The output should not be interpreted as a clinical diagnosis, causal conclusion, or treatment recommendation.
