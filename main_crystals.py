from pathlib import Path
import json
import pandas as pd
from polynet.train.evaluate_model import get_metrics, plot_learning_curves, plot_results
from polynet.train.train_gnn import train_GNN_ensemble
from crystals_experiment.crystal_graph import crystal_graph_dataset
from polynet.app.services.model_training import save_gnn_model
from polynet.utils.split_data import get_data_split_indices
from polynet.predict.predict_gnn import get_predictions_df_gnn

def main():

    general_dir = Path("crystals_experiment")

    targets = pd.read_csv(general_dir / 'data/id_prop.csv', header=None)

    print(targets)

    dataset = crystal_graph_dataset(
        root= general_dir / "data",
        max_d=3.5,
        step=0.5,
        vor_cut_off=3.5,
        targets=targets
    )

    print(dataset[0])

    targets = targets.rename(columns={"0": "id", "1": "target"})

    train_val_test_idxs = get_data_split_indices(
        data=targets.copy().set_index(0),
        split_type="train_val_test",
        n_bootstrap_iterations=5,
        val_ratio=0.1,
        test_ratio=0.1,
        target_variable_col="target",
        split_method="Random",
        train_set_balance=None,
        random_seed=20252025,
    )

    results_dir = general_dir / "results"

    results_dir.mkdir(exist_ok=True)

    gnn_conv_layers = {}
    gnn_conv_layers["GCN"] = {}
    gnn_conv_layers["GraphSAGE"] = {}
    gnn_conv_layers["TransformerConvGNN"] = {}
    gnn_conv_layers["GAT"] = {}
    gnn_conv_layers["MPNN"] = {}
    gnn_conv_layers["CGGNN"] = {}

    gnn_models, loaders = train_GNN_ensemble(
        experiment_path=results_dir,
        dataset=dataset,
        split_indexes=train_val_test_idxs,
        gnn_conv_params=gnn_conv_layers,
        problem_type="regression",
        num_classes=1,
        random_seed=20252025,
    )

    gnn_models_dir = results_dir / "models"
    gnn_models_dir.mkdir(exist_ok=True)
    for model_name, model in gnn_models.items():
        save_path = gnn_models_dir / f"{model_name}.pt"
        save_gnn_model(model, save_path)

    predictions_gnn = get_predictions_df_gnn(
        models=gnn_models,
        loaders=loaders,
        problem_type="regression",
        split_type="train_val_test",
        target_variable_name="target",
    )

    predictions_gnn.to_csv(results_dir / "predictions.csv")

    metrics_gnn = get_metrics(
        predictions=predictions_gnn,
        split_type="train_val_test",
        target_variable_name="target",
        trained_models=gnn_models.keys(),
        problem_type="regression",
    )

    gnn_metrics_path = results_dir / "metrics.json"
    with open(gnn_metrics_path, "w") as f:
        json.dump(metrics_gnn, f, indent=4)

    gnn_plots_dir = results_dir / "plots"
    gnn_plots_dir.mkdir(exist_ok=True)

    plot_learning_curves(models=gnn_models, save_path=gnn_plots_dir)

    plot_results(
        predictions=predictions_gnn,
        split_type="train_val_test",
        target_variable_name="target",
        ml_algorithms=gnn_models.keys(),
        problem_type="regression",
        save_path=gnn_plots_dir,
        class_names=None,
    )

if __name__ == "__main__":
    main()