import os
import sys
import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from omegaconf import OmegaConf
import time
# import db
from datetime import datetime
from inference import load_model, get_example,  inference

COMMENT_TEMPLATE_MD = """{} - {}
> {}"""


def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")


root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(root_dir))

## inference
cfg = OmegaConf.load(os.path.join(root_dir, "conf.yaml"))
model = load_model(cfg)


st.title("Welcome to Doctor DeepOr!")
st.markdown(
    "DeepOr is a diagnosis system developed by the team of [HILAB](http://www.healthinformaticslab.org). It orchestrates the deep learning's **representation capability** and conventional machine learning's **interpretability** into an end-to-end framework. This framework not only delivers better clinical prediction models, but also explicitly report the biomedical meaningful **biomarkers** for the prediction tasks. If you have any questions, please contact [Ruochi Zhang](mailto:zrc720@gmail.com) or [Fengfeng Zhou](mailto:FengfengZhou@gmail.com)")

uploaded_file = st.file_uploader("Upload clinical Data for a patient")


# IMPORTANT: Cache the conversion to prevent computation on every rerun
with open(cfg.data.example, "rb") as f:
    bytes = f.read()
    st.download_button(
        label="Download template",
        data=bytes,
        file_name='patient_example.xlsx',
        mime='text/xlsx',
    )

if uploaded_file is not None:
    try: 
        bytes_data = uploaded_file.read()
        with open("./{}".format(uploaded_file.name), "wb") as f:
            f.write(bytes_data)
        ts_tensor, demo_tensor, demo_data = get_example(
            uploaded_file.name, uploaded_file.name)
        res = inference(cfg, model, ts_tensor, demo_tensor)

        space(2)

        mortality = ["False", "True"]

        st.subheader("Prediction")
        st.success(
            "in-hospital mortality: {}, Probablity: {}".format(mortality[res["label"]], res["prob"]))

        # loading report
        
        with st.spinner('Loading for report...'):
            time.sleep(2)

        space(2)

        # display ICU data
        st.subheader("ICU stay for the predicted patient")
        chart_data = pd.DataFrame(
            data=ts_tensor.squeeze(0).cpu().numpy(),
            columns=cfg.model.ts_header
        )
        st.line_chart(chart_data)

        space(2)
        
        # display Demographic data
        st.subheader("Demographic data for the predicted patient")
        table_data = pd.DataFrame(
            data=[demo_data[1:]],
            columns=cfg.model.demo_feature_names
        )
        st.table(table_data)

        space(2)


        # display Important Factors
        st.subheader("Changes in important Factors")
        fs_index = res["index"]
        factor_data = chart_data.iloc[:, fs_index]
        st.area_chart(factor_data)
    except:
        st.error("Please Check Your Uploaded Files")