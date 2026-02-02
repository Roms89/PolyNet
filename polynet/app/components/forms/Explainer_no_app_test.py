from pathlib import Path

import pandas as pd
from rdkit.Chem import Descriptors
import streamlit as st
from torch_geometric.data import Dataset

from polynet.app.options.data import DataOptions
from polynet.app.options.state_keys import ExplainModelStateKeys, ProjectionPlotStateKeys
from polynet.app.services.descriptors import (
    calculate_rdkit_df_dict,
    get_unique_smiles,
    merge_weighted,
)
from polynet.app.services.explain_model import analyse_graph_embeddings, explain_model
from polynet.app.services.model_training import load_gnn_model
from polynet.app.utils import extract_number
from polynet.options.col_names import get_predicted_label_column_name, get_true_label_column_name
from polynet.options.enums import (
    DataSets,
    DimensionalityReduction,
    ExplainAlgorithms,
    ImportanceNormalisationMethods,
    ProblemTypes,
    Results,
)


model_dir = "crystals_experiment/results/models"

model_name = CGCNN_1.pt

model_path = model_dir / model_name
model = load_gnn_model(model_path)
