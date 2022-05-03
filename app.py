#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/AAAI/DeepOchestration/app/app.py
# Project: /data/zhangruochi/projects/AAAI/DeepOchestration/app
# Created Date: Tuesday, May 3rd 2022, 11:48:03 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Wed May 04 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
# 
# MIT License
# 
# Copyright (c) 2022 Silexon Ltd
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import os
import sys
import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from omegaconf import OmegaConf
import time
import db
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