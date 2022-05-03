#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/AAAI/DeepOchestration/inference.py
# Project: /data/zhangruochi/projects/AAAI/DeepOchestration/app
# Created Date: Tuesday, May 3rd 2022, 4:58:11 pm
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
import mlflow
from omegaconf import OmegaConf
from preprocess import get_example
import torch

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(root_dir))


def load_model(cfg):
    model_path = os.path.join(root_dir, cfg.model.model_path)
    ## loading table_classifier
    # print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model.eval()

    return model

def inference(cfg, model, icu_data, demo_data):

    with torch.no_grad():
        output = model(icu_data, demo_data)

    pre_logits, (x, ae_rep, recon_x, fs_rep, f_index, mu, logvar) = output

    pre_probas = torch.sigmoid(pre_logits).squeeze(-1)
    pre_labels = torch.round(pre_probas)

    res = {"prob": round(pre_probas.item(),4), "label": int(pre_labels.item()), "index": f_index.cpu().numpy().tolist()}

    return res


if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(root_dir, "conf.yaml"))
    model = load_model(cfg)
    # input = get_example("./ts.csv", "./demo.csv")
    ts_tensor, demo_tensor, demo_data = get_example("icu_patient_1.xlsx", "icu_patient_1.xlsx")    
    res = inference(cfg, model, ts_tensor, demo_tensor)
    print(res)