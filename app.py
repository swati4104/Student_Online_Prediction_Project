import streamlit as st
with st.form(key="predict_form"):
st.write("Enter values for features (or upload rows to predict on)")
# Create inputs based on dtype
input_data = {}
cols_for_inputs = feature_columns
for c in cols_for_inputs:
series = df[c] if df is not None and c in df.columns else None
if series is not None and pd.api.types.is_numeric_dtype(series):
min_v = float(series.min()) if not np.isnan(series.min()) else 0.0
max_v = float(series.max()) if not np.isnan(series.max()) else min_v + 1.0
mean_v = float(series.mean()) if not np.isnan(series.mean()) else 0.0
input_data[c] = st.number_input(label=f"{c}", value=mean_v, format="%f")
elif series is not None and pd.api.types.is_categorical_dtype(series) or series is not None and series.dtype == object:
uniques = list(series.dropna().unique())[:200]
default = uniques[0] if uniques else ""
input_data[c] = st.selectbox(label=f"{c}", options=uniques, index=0 if uniques else None)
else:
# fallback
input_data[c] = st.text_input(label=f"{c}", value="")


uploaded_rows = st.file_uploader("(Optional) Upload CSV with same feature columns for batch predictions", type=["csv"])
submit = st.form_submit_button("Predict")


if submit:
# prepare DataFrame X
if uploaded_rows is not None:
try:
X_new = pd.read_csv(uploaded_rows)
missing = [c for c in feature_columns if c not in X_new.columns]
if missing:
st.error(f"Uploaded CSV is missing columns: {missing}")
else:
X = X_new[feature_columns]
except Exception as e:
st.error(f"Failed to read uploaded CSV: {e}")
X = pd.DataFrame([input_data])
else:
X = pd.DataFrame([input_data], columns=feature_columns)


st.write("Prepared input for prediction:")
st.dataframe(X)


if model is None:
st.error("No model loaded to make predictions.")
else:
preds = model_predict(model, X)
result_df = X.reset_index(drop=True).join(preds.reset_index(drop=True))
st.success("Prediction completed")
st.dataframe(result_df)


# allow download
csv = result_df.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
else:
st.info("Select features from the dataset first")


# Footer / troubleshooting
st.markdown("---")
st.info("If your model expects specific preprocessing (scaler, encoder), ensure those preprocessing steps are part of the saved object (Pipeline). If predictions fail, try saving/loading a `sklearn.pipeline.Pipeline` containing both preprocessing and estimator.")

