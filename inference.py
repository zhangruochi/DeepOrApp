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