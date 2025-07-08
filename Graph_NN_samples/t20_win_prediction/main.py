import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim

# Import from our local files
from data_builder import build_dataset
from models import build_logistic_regression_pipeline, TeamGCN, build_normalized_adj

# Set styling for premium publication-quality plots
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Color palette for consistent aesthetics
PRIMARY_COLOR = "#1f77b4" # Slate Blue
SECONDARY_COLOR = "#ff7f0e" # Coral
DARK_BG = "#1e1e24"

def run_comparison():
    print("=" * 60)
    print("   T20 WORLD CUP MATCH OUTCOME PREDICTION COMPARISON   ")
    print("          GRAPH NEURAL NETWORK vs. LOGISTIC REGRESSION ")
    print("=" * 60)
    
    # -----------------------------------------------------------------
    # STEP 1: LOAD OR GENERATE DATASET
    # -----------------------------------------------------------------
    data_path = "t20_world_cup_matches.csv"
    if not os.path.exists(data_path):
        print("[1/5] Dataset CSV not found. Generating realistic historical dataset...")
        df = build_dataset(data_path)
    else:
        print("[1/5] Dataset CSV found. Loading...")
        df = pd.read_csv(data_path)
        # Check for NaNs (e.g., if an older incorrect version of the CSV is cached)
        if df.isna().any().any():
            print("⚠️ Detected missing values (NaNs) in the loaded dataset. Re-generating fresh clean dataset...")
            df = build_dataset(data_path)
        
    print(f"Loaded {len(df)} matches across {df['tournament'].nunique()} T20 World Cup editions.")
    
    # -----------------------------------------------------------------
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & STATS
    # -----------------------------------------------------------------
    print("\n" + "-" * 40)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("-" * 40)
    
    overall_bat_first_win = df["batting_first_won"].mean() * 100
    print(f"Overall Win Rate for Team Batting First: {overall_bat_first_win:.2f}%")
    print(f"Overall Win Rate for Team Batting Second: {(100 - overall_bat_first_win):.2f}%")
    
    print("\nWin Rate by Tournament Edition:")
    tourney_stats = df.groupby("tournament")["batting_first_won"].agg(["count", "mean"])
    tourney_stats["mean"] *= 100
    tourney_stats = tourney_stats.rename(columns={"count": "Matches", "mean": "Bat First Win %"})
    print(tourney_stats.to_string())
    
    print("\nWin Rate by Select High-Impact Venues:")
    venue_stats = df.groupby("venue")["batting_first_won"].agg(["count", "mean"])
    venue_stats["mean"] *= 100
    venue_stats = venue_stats.rename(columns={"count": "Matches", "mean": "Bat First Win %"})
    # Filter venues with 5 or more matches for statistical significance
    significant_venues = venue_stats[venue_stats["Matches"] >= 5].sort_values("Bat First Win %", ascending=False)
    print(significant_venues.to_string())
    
    # -----------------------------------------------------------------
    # STEP 3: PREPROCESS & ENCODE FEATURES
    # -----------------------------------------------------------------
    # Create unique team mappings
    all_teams = sorted(list(set(df["team_1"].unique()).union(set(df["team_2"].unique()))))
    team_to_idx = {team: idx for idx, team in enumerate(all_teams)}
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}
    num_teams = len(all_teams)
    
    df["team_1_idx"] = df["team_1"].map(team_to_idx)
    df["team_2_idx"] = df["team_2"].map(team_to_idx)
    
    # Feature Engineering: Toss Winner advantage
    # 1 if Team 1 (batting first) won the toss, 0 if Team 2 (batting second) won the toss
    df["team_1_won_toss"] = (df["toss_winner"] == df["team_1"]).astype(int)
    
    # Rating Difference
    df["rank_diff"] = df["team_2_rank"] - df["team_1_rank"]
    
    # Train/Test Split (80% Train, 20% Test)
    # Using stratify to ensure target distribution is identical in splits
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["batting_first_won"])
    
    # -----------------------------------------------------------------
    # STEP 4: LOGISTIC REGRESSION TRAINING
    # -----------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[2/5] TRAINING LOGISTIC REGRESSION BASELINE")
    print("-" * 40)
    
    numerical_cols = ["team_1_rank", "team_2_rank", "rank_diff"]
    categorical_cols = ["venue", "team_1", "team_2"]
    binary_cols = ["team_1_won_toss", "is_day_night", "dew_factor"]
    
    all_features = numerical_cols + categorical_cols + binary_cols
    
    X_train_lr = train_df[all_features]
    y_train_lr = train_df["batting_first_won"]
    X_test_lr = test_df[all_features]
    y_test_lr = test_df["batting_first_won"]
    
    lr_pipeline = build_logistic_regression_pipeline(numerical_cols, categorical_cols)
    lr_pipeline.fit(X_train_lr, y_train_lr)
    
    # Evaluation
    lr_preds = lr_pipeline.predict(X_test_lr)
    lr_probs = lr_pipeline.predict_proba(X_test_lr)[:, 1]
    
    lr_acc = accuracy_score(y_test_lr, lr_preds)
    lr_precision, lr_recall, lr_f1, _ = precision_recall_fscore_support(y_test_lr, lr_preds, average='binary')
    lr_auc = roc_auc_score(y_test_lr, lr_probs)
    
    print(f"Logistic Regression Test Accuracy: {lr_acc*100:.2f}%")
    print(f"Logistic Regression Test ROC-AUC : {lr_auc:.4f}")
    print(f"Logistic Regression Test F1-Score: {lr_f1:.4f}")
    
    # -----------------------------------------------------------------
    # STEP 5: GRAPH NEURAL NETWORK (GNN) TRAINING
    # -----------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[3/5] PREPARING GRAPH NEURAL NETWORK (PURE PYTORCH)")
    print("-" * 40)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for PyTorch GNN: {device}")
    
    # Build match-specific edge features (one-hot encode venue + toss, day_night, dew)
    venue_dummies = pd.get_dummies(df["venue"], prefix="venue").astype(float)
    venue_cols = list(venue_dummies.columns)
    
    edge_feats_df = pd.concat([
        df["team_1_won_toss"].astype(float),
        df["is_day_night"].astype(float),
        df["dew_factor"].astype(float),
        venue_dummies
    ], axis=1)
    
    edge_features_all = torch.tensor(edge_feats_df.values, dtype=torch.float32, device=device)
    team_1_indices_all = torch.tensor(df["team_1_idx"].values, dtype=torch.long, device=device)
    team_2_indices_all = torch.tensor(df["team_2_idx"].values, dtype=torch.long, device=device)
    labels_all = torch.tensor(df["batting_first_won"].values, dtype=torch.float32, device=device)
    
    # Get train/test splits for tensors
    train_indices = train_df.index.values
    test_indices = test_df.index.values
    
    # 1. Build Adjacency Matrix using ONLY training matches to prevent data leakage!
    norm_adj_train = build_normalized_adj(num_teams, train_df, device)
    
    # GCN parameters
    embed_dim = 16
    hidden_dim = 16
    out_dim = 8
    num_edge_features = edge_features_all.shape[1]
    
    model = TeamGCN(num_teams, embed_dim, hidden_dim, out_dim, num_edge_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-3)
    
    # Extract training inputs
    train_t1 = team_1_indices_all[train_indices]
    train_t2 = team_2_indices_all[train_indices]
    train_edge_feat = edge_features_all[train_indices]
    train_labels = labels_all[train_indices]
    
    # Extract testing inputs
    test_t1 = team_1_indices_all[test_indices]
    test_t2 = team_2_indices_all[test_indices]
    test_edge_feat = edge_features_all[test_indices]
    test_labels = labels_all[test_indices]
    
    print("Training GNN model over 250 epochs...")
    
    epochs = 250
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on train graph
        logits = model(norm_adj_train, train_t1, train_t2, train_edge_feat)
        loss = criterion(logits, train_labels)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            # Eval train metrics
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == train_labels).float().mean().item()
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {acc*100:.2f}%")
            
    # Evaluation on Test Set
    model.eval()
    with torch.no_grad():
        # Evaluate on test set using the train-built adjacency matrix (inductive evaluation)
        test_logits = model(norm_adj_train, test_t1, test_t2, test_edge_feat)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        
    y_test_gnn = test_labels.cpu().numpy()
    gnn_acc = accuracy_score(y_test_gnn, test_preds)
    gnn_precision, gnn_recall, gnn_f1, _ = precision_recall_fscore_support(y_test_gnn, test_preds, average='binary')
    gnn_auc = roc_auc_score(y_test_gnn, test_probs)
    
    print("\n" + "-" * 40)
    print("GNN TRAINING COMPLETED")
    print("-" * 40)
    print(f"PyTorch GNN Test Accuracy: {gnn_acc*100:.2f}%")
    print(f"PyTorch GNN Test ROC-AUC : {gnn_auc:.4f}")
    print(f"PyTorch GNN Test F1-Score: {gnn_f1:.4f}")
    
    # -----------------------------------------------------------------
    # STEP 6: COMPARATIVE REPORT
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("                 MODEL COMPARISON REPORT                 ")
    print("=" * 60)
    comparison_table = pd.DataFrame({
        "Metric": ["Accuracy", "ROC-AUC", "F1-Score", "Precision", "Recall"],
        "Logistic Regression": [lr_acc, lr_auc, lr_f1, lr_precision, lr_recall],
        "Graph Neural Network (GCN)": [gnn_acc, gnn_auc, gnn_f1, gnn_precision, gnn_recall]
    })
    # Format values as percentages or rounded floats
    for col in ["Logistic Regression", "Graph Neural Network (GCN)"]:
        comparison_table[col] = comparison_table.apply(
            lambda row: f"{row[col]*100:.2f}%" if row["Metric"] in ["Accuracy", "Precision", "Recall"] else f"{row[col]:.4f}",
            axis=1
        )
    print(comparison_table.to_string(index=False))
    
    # Explanation
    print("\nAnalytical Insight:")
    if gnn_auc > lr_auc:
        print("💡 The Graph Neural Network outperformed Logistic Regression!")
        print("   This is because the GNN leverages the graph structure, allowing it to capture transitive")
        print("   team relationships (e.g., if Team A beat Team B, and Team B beat Team C, the GNN infers")
        print("   strengths transitively). Logistic regression, treating each match as a separate independent")
        print("   row, cannot model this relational tournament topology.")
    else:
        print("💡 Both models performed highly competitively!")
        print("   Logistic Regression is a strong baseline because it directly weighs features like rank difference")
        print("   and venue biases. The GNN matches this performance while learning custom team embeddings")
        print("   from scratch based strictly on the tournament's match schedule graph.")
        
    # -----------------------------------------------------------------
    # STEP 7: PREMIUM PERFORMANCE DASHBOARD & TEAM EMBEDDINGS PLOT
    # -----------------------------------------------------------------
    print("\n[4/5] Extracting learned GNN Team Embeddings...")
    model.eval()
    with torch.no_grad():
        all_nodes = torch.arange(num_teams, device=device)
        final_team_embeds = model.team_embeddings(all_nodes).cpu().numpy()
        
    # Perform PCA on embeddings to project to 2D
    pca = PCA(n_components=2)
    embeds_2d = pca.fit_transform(final_team_embeds)
    
    # Calculate each team's actual win rate in the dataset for coloring the embedding plot
    team_win_rates = {}
    for team in all_teams:
        team_matches = df[(df["team_1"] == team) | (df["team_2"] == team)]
        team_wins = df[df["winner"] == team]
        rate = len(team_wins) / len(team_matches) if len(team_matches) > 0 else 0
        team_win_rates[team] = rate
        
    win_rates_vector = np.array([team_win_rates[idx_to_team[i]] for i in range(num_teams)])
    
    print("[5/5] Generating premium comparison dashboard in 'plots/performance_comparison.png'...")
    os.makedirs("plots", exist_ok=True)
    
    # Plotting Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. ROC Curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test_lr, lr_probs)
    gnn_fpr, gnn_tpr, _ = roc_curve(y_test_gnn, test_probs)
    
    axs[0, 0].plot(lr_fpr, lr_tpr, color=SECONDARY_COLOR, lw=3, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    axs[0, 0].plot(gnn_fpr, gnn_tpr, color=PRIMARY_COLOR, lw=3, label=f'PyTorch GNN (AUC = {gnn_auc:.3f})')
    axs[0, 0].plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    axs[0, 0].set_xlim([0.0, 1.0])
    axs[0, 0].set_ylim([0.0, 1.05])
    axs[0, 0].set_xlabel('False Positive Rate', fontweight='bold')
    axs[0, 0].set_ylabel('True Positive Rate', fontweight='bold')
    axs[0, 0].set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', pad=10)
    axs[0, 0].legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    
    # 2. Confusion Matrices
    lr_cm = confusion_matrix(y_test_lr, lr_preds)
    gnn_cm = confusion_matrix(y_test_gnn, test_preds)
    
    # Flatten axs for heatmaps
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axs[0, 1].set_title('Confusion Matrix: Logistic Regression', fontweight='bold', pad=10)
    axs[0, 1].set_xlabel('Predicted Label', fontweight='bold')
    axs[0, 1].set_ylabel('True Label', fontweight='bold')
    axs[0, 1].set_xticklabels(['Bat 2nd Win (0)', 'Bat 1st Win (1)'])
    axs[0, 1].set_yticklabels(['Bat 2nd Win (0)', 'Bat 1st Win (1)'], rotation=0)
    
    sns.heatmap(gnn_cm, annot=True, fmt='d', cmap='Oranges', ax=axs[1, 0], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axs[1, 0].set_title('Confusion Matrix: PyTorch Graph Neural Network', fontweight='bold', pad=10)
    axs[1, 0].set_xlabel('Predicted Label', fontweight='bold')
    axs[1, 0].set_ylabel('True Label', fontweight='bold')
    axs[1, 0].set_xticklabels(['Bat 2nd Win (0)', 'Bat 1st Win (1)'])
    axs[1, 0].set_yticklabels(['Bat 2nd Win (0)', 'Bat 1st Win (1)'], rotation=0)
    
    # 3. GCN Training Loss Curve
    axs[1, 1].plot(loss_history, color='#2ca02c', lw=2.5, label='BCE Loss')
    axs[1, 1].set_xlabel('Epochs', fontweight='bold')
    axs[1, 1].set_ylabel('BCE Loss', fontweight='bold')
    axs[1, 1].set_title('PyTorch GNN: Training Convergence Curve', fontweight='bold', pad=10)
    # Add final loss annotation
    axs[1, 1].annotate(f'Final Loss: {loss_history[-1]:.4f}', 
                        xy=(epochs-1, loss_history[-1]), 
                        xytext=(epochs-80, loss_history[-1] + 0.15),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    
    # Create an inset/ax replacement for 4th subplot: Team Embeddings Projection
    # We will replace axs[1,1] or let's create a 5th or overlay it. 
    # Actually, we can use axs[1,1] for loss, and we will place the Team Embeddings inside the 4th quadrant (replacing axs[1,1] with loss, and moving things).
    # Wait, the grid has 2x2 = 4 axes: axs[0,0], axs[0,1], axs[1,0], axs[1,1].
    # Let's adjust axs[1,1] to show GNN Team Embeddings, which is much more premium and visually stunning! 
    # And we can overlay the GNN loss curve inside it as a small inset or just plot the embeddings in the 4th axis since embeddings are so visual.
    # Yes! Let's plot the GNN Team Embeddings in axs[1,1] and make the loss curve an inset inside it, or vice versa. Let's make axs[1,1] the GNN Team Embeddings, which is extremely unique!
    axs[1, 1].clear() # Clear the loss curve from axs[1,1] to plot the team embeddings
    
    scatter = axs[1, 1].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=win_rates_vector, cmap='coolwarm', s=250, edgecolors='black', zorder=3)
    cbar = fig.colorbar(scatter, ax=axs[1, 1])
    cbar.set_label('Actual Team Win Rate in Dataset', fontweight='bold')
    
    # Add annotations for each team
    for i, team in enumerate(all_teams):
        axs[1, 1].annotate(team, (embeds_2d[i, 0], embeds_2d[i, 1]), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center', 
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.85))
        
    axs[1, 1].set_title('Learned GNN Team Embeddings Projection (2D PCA)', fontweight='bold', pad=10)
    axs[1, 1].set_xlabel('Latent Factor 1', fontweight='bold')
    axs[1, 1].set_ylabel('Latent Factor 2', fontweight='bold')
    
    # Now let's place the GNN loss curve inside axs[1,1] as a small floating inset plot! This is an extremely premium layout!
    # Inset location: upper left of axs[1,1]
    inset_ax = axs[1, 1].inset_axes([0.05, 0.05, 0.35, 0.28])
    inset_ax.plot(loss_history, color='#2ca02c', lw=1.5)
    inset_ax.set_title('GNN Train Loss', fontsize=9, fontweight='bold')
    inset_ax.set_xticks([])
    inset_ax.tick_params(labelsize=8)
    
    plt.suptitle("Model Performance Comparison: Graph Neural Network vs. Logistic Regression\nICC T20 World Cup Match Win Predictions (Batting First vs. Batting Second)", fontweight='bold', y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plot_output_path = os.path.join("plots", "performance_comparison.png")
    plt.savefig(plot_output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard saved successfully to {plot_output_path}!")
    print("=" * 60)
    print("SUCCESS: Model training, evaluation, and dashboard visualization complete.")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()
