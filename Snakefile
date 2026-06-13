VARIANTS = ["baseline", "embedding", "attention", "full"]
EPOCHS = 30


rule all:
    input:
        expand("results/{variant}_test_results.csv", variant=VARIANTS),
        expand("results/{variant}_history.csv", variant=VARIANTS),
        expand("results/{variant}_test_metrics.csv", variant=VARIANTS),
        expand("models/{variant}.pt", variant=VARIANTS),
        "results/model_comparison.csv",
        "results/cell_type_metrics.csv",
        "figures/validation_loss_curves.png",
        "figures/model_mse_comparison.png",
        "figures/model_rmse_comparison.png",
        "figures/model_mae_comparison.png",
        "figures/model_pearson_comparison.png",
        "figures/model_spearman_comparison.png",
        "figures/model_r2_comparison.png",
        "figures/embedding_predicted_vs_true.png",
        "figures/cell_type_pearson_comparison.png",
        "figures/cell_type_r2_comparison.png",
        "figures/HepG2_predicted_vs_true.png",
        "figures/K562_predicted_vs_true.png",
        "figures/WTC11_predicted_vs_true.png"


rule preprocess:
    input:
        fasta="data/raw/ENCFF795PSI.fasta",
        hepg2="data/raw/ENCFF009HVM_HepG2.tsv",
        k562="data/raw/ENCFF068BWG_K562.tsv",
        wtc11="data/raw/ENCFF357GSY_WTC11.tsv"
    output:
        data="data/processed/data.csv",
        train="data/processed/train.csv",
        val="data/processed/val.csv",
        test="data/processed/test.csv"
    shell:
        "python src/preprocess.py"


rule train_model:
    input:
        train="data/processed/train.csv",
        val="data/processed/val.csv",
        test="data/processed/test.csv"
    output:
        result="results/{variant}_test_results.csv",
        history="results/{variant}_history.csv",
        metrics="results/{variant}_test_metrics.csv",
        model="models/{variant}.pt"
    params:
        epochs=EPOCHS
    shell:
        "python src/train.py --variant {wildcards.variant} --epochs {params.epochs}"


rule plots:
    input:
        expand("results/{variant}_test_results.csv", variant=VARIANTS),
        expand("results/{variant}_history.csv", variant=VARIANTS)
    output:
        comparison="results/model_comparison.csv",
        validation="figures/validation_loss_curves.png",
        mse="figures/model_mse_comparison.png",
        rmse="figures/model_rmse_comparison.png",
        mae="figures/model_mae_comparison.png",
        pearson="figures/model_pearson_comparison.png",
        spearman="figures/model_spearman_comparison.png",
        r2="figures/model_r2_comparison.png",
        scatter="figures/embedding_predicted_vs_true.png"
    shell:
        "python src/plots.py"


rule cell_type_analysis:
    input:
        predictions="results/embedding_test_results.csv",
        test_data="data/processed/test.csv"
    output:
        metrics="results/cell_type_metrics.csv",
        pearson="figures/cell_type_pearson_comparison.png",
        r2="figures/cell_type_r2_comparison.png",
        hepg2="figures/HepG2_predicted_vs_true.png",
        k562="figures/K562_predicted_vs_true.png",
        wtc11="figures/WTC11_predicted_vs_true.png"
    shell:
        "python src/analyze_by_cell_type.py"
