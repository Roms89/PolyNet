from pathlib import Path
import json
import pandas as pd
from polynet.train.evaluate_model import get_metrics, plot_learning_curves, plot_results
from crystals_experiment.crystal_graph import crystal_graph_dataset
from polynet.app.services.model_training import save_gnn_model
from polynet.utils.split_data import get_data_split_indices
from polynet.predict.predict_gnn import get_predictions_df_gnn
from polynet.call_methods import (
    compute_class_weights,
    create_network,
    make_loss,
    make_optimizer,
    make_scheduler,
)
from polynet.options.enums import NetworkParams, Networks, Optimizers, ProblemTypes, Schedulers
from polynet.options.search_grids import get_grid_search
from polynet.app.services.predict_model import predict_unseen_gnn
import torch
from torch_geometric.loader import DataLoader

def main():

    test_dir = Path("crystals_test")

    model_dir = "crystals_experiment/results/models"

    targets = pd.read_csv(test_dir / 'data/id_prop.csv', header=None)

    print(targets)

    dataset = crystal_graph_dataset(
        root=test_dir/"data",
        max_d=3.5,
        step=0.5,
        vor_cut_off=3.5,
        targets=targets
        )

    print(dataset[0])

    targets = targets.rename(columns={"0": "id", "1": "target"})

    test_loader = DataLoader(dataset, shuffle=False)

    loaders = ([], [], test_loader)

    results_dir = test_dir / "results"

    results_dir.mkdir(exist_ok=True)

    gnn_models = {}

    model_path = Path(model_dir)

    for model in model_path.iterdir():
        if model.is_file():
            gnn_models.update({model.stem:torch.load(model,
                                                  map_location=torch.device('cpu'), weights_only=False)})


    predictions_gnn = predict_unseen_gnn(
        models=gnn_models,
        dataset=dataset,
        target_variable_name="target",
        problem_type="regression",
        )

    print(f"{model}: ", predictions_gnn)

    predictions_gnn.to_csv(results_dir / "predictions.csv")

if __name__ == "__main__":
    main()
