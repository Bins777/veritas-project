import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from graphdatascience import GraphDataScience
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import csv
import os

# Computational analysis output path (relative to working directory)
COMP_CSV_PATH = os.path.join("computational_analysis_outputs", "timing_results.csv")
os.makedirs(os.path.dirname(COMP_CSV_PATH), exist_ok=True)

def append_timing_row(model_name, train_sec, infer_sec, n_test, hardware):
    per_sample_ms = (infer_sec / n_test) * 1000.0
    file_exists = os.path.isfile(COMP_CSV_PATH)
    with open(COMP_CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Training_Time_Sec', 'Inference_Time_Sec',
                             'Per_Sample_Inference_Ms', 'Hardware'])
        writer.writerow([model_name,
                         f"{train_sec:.4f}",
                         f"{infer_sec:.4f}",
                         f"{per_sample_ms:.4f}",
                         hardware])
    print(f"📝 Timing appended: {model_name} | {hardware}")

# ============================================================
# EXPERIMENT CONFIGURATION
# Change EXPERIMENT to switch between runs
# 'full' = jurisdiction + structural + risk
# 'structural' = structural + risk only (no jurisdiction)
# ============================================================
EXPERIMENT = 'full' # Options: 'full' or 'structural'

if EXPERIMENT == 'full':
    OUTPUT_DIR = "HGT_main_outputs"
    ENTITY_FEATURES = ['f1_secrecy', 'f2_facilitation', 'f3_governance',
                       'in_degree', 'out_degree', 'is_risk']
    OFFICER_FEATURES = ['f1_secrecy', 'f2_facilitation', 'f3_governance',
                        'in_degree', 'out_degree', 'is_risk']
    INTERMEDIARY_FEATURES = ['f1_secrecy', 'f2_facilitation', 'f3_governance',
                             'in_degree', 'out_degree', 'is_bridge',
                             'is_global_hub', 'is_risk']
    EXPERIMENT_LABEL = 'HGT Full Features'
else:
    OUTPUT_DIR = "HGT_structural_only_outputs"
    ENTITY_FEATURES = ['in_degree', 'out_degree', 'is_risk']
    OFFICER_FEATURES = ['in_degree', 'out_degree', 'is_risk']
    INTERMEDIARY_FEATURES = ['in_degree', 'out_degree', 'is_bridge',
                             'is_global_hub', 'is_risk']
    EXPERIMENT_LABEL = 'HGT Structural Only (No Jurisdiction)'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cycle test output folder, separate from main test outputs
if EXPERIMENT == 'full':
    CYCLE_OUTPUT_DIR = "HGT_cycle_zeroshot_full_outputs"
else:
    CYCLE_OUTPUT_DIR = "HGT_cycle_zeroshot_structural_outputs"

os.makedirs(CYCLE_OUTPUT_DIR, exist_ok=True)

# ============================================================
# PHASE 1: CONNECTION & RISK LOOKUP
# ------------------------------------------------------------
# Connection details and the dataset are NOT bundled with this
# public release. Credentials are read from the environment, and
# the underlying graph/jurisdiction data is available on request
# from the Veritas team (see README).
#   NEO4J_URI       (default: neo4j://127.0.0.1:7687)
#   NEO4J_USER      (default: neo4j)
#   NEO4J_PASSWORD  (required)
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
risk_lookup = pd.read_csv('jurisdiction_risk.csv')

data = HeteroData()

# ============================================================
# PHASE 2: DATA ENGINE — FULL FEATURE MATRIX & EDGE LOADING
# ============================================================

# ------------------------------------------------------------
# 2A. ENTITY NODES
# Features: f1, f2, f3 (jurisdiction risk)
#           in_degree, out_degree, is_anomaly
# Label: is_shell
# ------------------------------------------------------------
entity_query = """
    MATCH (e:Entity)
    RETURN id(e) AS node_id,
           e.is_shell AS label,
           e.jurisdiction AS jurisdiction,
           COALESCE(e.in_degree, 0) AS in_degree,
           COALESCE(e.out_degree, 0) AS out_degree,
           CASE WHEN e.is_risk = true THEN 1 ELSE 0 END AS is_risk,
           COALESCE(e.is_test_only, 0) AS is_test_only
"""
entity_df = gds.run_cypher(entity_query)
entity_df = entity_df.merge(risk_lookup, on='jurisdiction', how='left').fillna(0)

entity_map = {old: i for i, old in enumerate(entity_df['node_id'])}

data['Entity'].x = torch.tensor(
    entity_df[ENTITY_FEATURES].values,
    dtype=torch.float
)
data['Entity'].y = torch.tensor(entity_df['label'].values, dtype=torch.long)
print(f"✅ Entity nodes loaded: {data['Entity'].num_nodes} nodes, {data['Entity'].x.shape[1]} features")

# Debug: confirm cycle entity degrees as loaded by GNN
cycle_check = entity_df[entity_df['is_test_only'] == 1]
print(f"\n Debug: {len(cycle_check)} cycle entities loaded by GNN")
print(f"   Cycle in_degree avg: {cycle_check['in_degree'].mean():.2f}")
print(f"   Cycle in_degree max: {cycle_check['in_degree'].max()}")
print(f"   Cycle out_degree avg: {cycle_check['out_degree'].mean():.2f}")
print(f"   Cycle is_risk values: {cycle_check['is_risk'].unique()}")
# ------------------------------------------------------------
# 2B. OFFICER NODES
# Features: f1, f2, f3 (jurisdiction risk)
#           entity_count, is_anomaly
# ------------------------------------------------------------
officer_query = """
    MATCH (o:Officer)
    RETURN id(o) AS node_id,
           o.jurisdiction AS jurisdiction,
           COALESCE(o.in_degree, 0) AS in_degree,
           COALESCE(o.out_degree, 0) AS out_degree,
           CASE WHEN o.is_risk = true THEN 1 ELSE 0 END AS is_risk
"""
officer_df = gds.run_cypher(officer_query)
officer_df = officer_df.merge(risk_lookup, on='jurisdiction', how='left').fillna(0)

officer_map = {old: i for i, old in enumerate(officer_df['node_id'])}

data['Officer'].x = torch.tensor(
    officer_df[OFFICER_FEATURES].values,
    dtype=torch.float
)
print(f"✅ Officer nodes loaded: {data['Officer'].num_nodes} nodes, {data['Officer'].x.shape[1]} features")

# ------------------------------------------------------------
# 2C. INTERMEDIARY NODES
# Features: f1, f2, f3 (jurisdiction risk)
#           entity_count, is_bridge, is_global_hub, is_anomaly
# ------------------------------------------------------------
intermediary_query = """
    MATCH (i:Intermediary)
    RETURN id(i) AS node_id,
           CASE 
               WHEN i.jurisdiction IS NULL THEN 'Unknown'
               WHEN size(i.jurisdiction) > 0 AND i.jurisdiction STARTS WITH '[' 
               THEN split(replace(replace(i.jurisdiction, '[', ''), ']', ''), ',')[0]
               ELSE i.jurisdiction 
           END AS jurisdiction,
           COALESCE(i.in_degree, 0) AS in_degree,
           COALESCE(i.out_degree, 0) AS out_degree,
           CASE WHEN i.is_bridge = true THEN 1 ELSE 0 END AS is_bridge,
           CASE WHEN i.is_global_hub = true THEN 1 ELSE 0 END AS is_global_hub,
           CASE WHEN i.is_risk = true THEN 1 ELSE 0 END AS is_risk
"""
intermediary_df = gds.run_cypher(intermediary_query)

# Step 1: Flatten any list values
intermediary_df['jurisdiction'] = intermediary_df['jurisdiction'].apply(
    lambda x: x[0].strip() if isinstance(x, list) 
    else str(x).strip() if x is not None 
    else 'Unknown'
)


# Step 2: Map known mismatches to CSV format
jurisdiction_map = {
    'GBR': 'United Kingdom',
    'UK': 'United Kingdom',
    'USA': 'United States',
    'US': 'United States',
    'UAE': 'United Arab Emirates',
    'HK': 'Hong Kong',
    'BVI': 'British Virgin Islands',
    'Cayman': 'Cayman Islands',
    'Cayman Islands, British West Indies': 'Cayman Islands',
    'Virgin Islands, British': 'British Virgin Islands',
    'Unknown': 'United Kingdom',
    'nan': 'United Kingdom',
    '': 'United Kingdom',
    'None': 'United Kingdom'
}
intermediary_df['jurisdiction'] = intermediary_df['jurisdiction'].replace(jurisdiction_map)

# Step 3: Now merge is safe
intermediary_df = intermediary_df.merge(risk_lookup, on='jurisdiction', how='left').fillna(0)


intermediary_map = {old: i for i, old in enumerate(intermediary_df['node_id'])}

data['Intermediary'].x = torch.tensor(
    intermediary_df[INTERMEDIARY_FEATURES].values,
    dtype=torch.float
)
print(f"✅ Intermediary nodes loaded: {data['Intermediary'].num_nodes} nodes, {data['Intermediary'].x.shape[1]} features")

# ------------------------------------------------------------
# 2D. EDGE LOADING WITH K-BARRIER AND FRICTION FACTOR
# Friction factor phi = 1.25 for cross-jurisdiction edges
# K-Barrier = 25 max connections per target node (Hu et al., 2020)
# ------------------------------------------------------------

def load_edge(query, src_map, dst_map, edge_key, apply_friction=False, apply_barrier=True):
    """
    Generic edge loader with optional friction factor and K-Barrier.
    
    Args:
        query: Cypher query returning source, s_jur, target, t_jur
        src_map: source node ID mapping
        dst_map: target node ID mapping  
        edge_key: PyG edge type tuple e.g. ('Officer', 'BENEFICIAL_OWNER', 'Entity')
        apply_friction: whether to apply cross-jurisdiction weight penalty
        apply_barrier: whether to apply K-Barrier cap of 25
    """
    df = gds.run_cypher(query)
    
    if df.empty:
        print(f" No edges found for {edge_key}")
        return

    # Filter to only edges where both nodes exist in maps
    df = df[df['source'].isin(src_map) & df['target'].isin(dst_map)]

    src_idx = [src_map[s] for s in df['source']]
    dst_idx = [dst_map[d] for d in df['target']]

    # Friction factor phi
    if apply_friction and 's_jur' in df.columns and 't_jur' in df.columns:
        weights = [
            1.25 if row['s_jur'] != row['t_jur'] else 1.0
            for _, row in df.iterrows()
        ]
    else:
        weights = [1.0] * len(src_idx)

    edges_df = pd.DataFrame({
        'source': src_idx,
        'target': dst_idx,
        'weight': weights
    })

    original_count = len(edges_df)

    # K-Barrier: cap at 25 connections per target node
    if apply_barrier:
        edges_df = edges_df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 25), random_state=42)
        )

    print(f"✂️ {edge_key[1]}: {original_count} → {len(edges_df)} edges after K-Barrier")

    data[edge_key].edge_index = torch.tensor(
        [edges_df['source'].tolist(), edges_df['target'].tolist()],
        dtype=torch.long
    )
    data[edge_key].edge_weight = torch.tensor(
        edges_df['weight'].tolist(),
        dtype=torch.float
    )

print("\n Loading edges with K-Barrier (max 25 per target) and Friction Factors...")

# 1. Entity CONNECTED_TO Officer (2657)
load_edge(
    """
    MATCH (a:Entity)-[:CONNECTED_TO]->(b:Officer)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    entity_map, officer_map,
    ('Entity', 'CONNECTED_TO', 'Officer'),
    apply_friction=True
)

# 2. Entity CONNECTED_TO Intermediary (2022)
load_edge(
    """
    MATCH (a:Entity)-[:CONNECTED_TO]->(b:Intermediary)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    entity_map, intermediary_map,
    ('Entity', 'CONNECTED_TO', 'Intermediary'),
    apply_friction=True
)

# 3. Officer OFFICER_OF Entity (1565)
load_edge(
    """
    MATCH (a:Officer)-[:OFFICER_OF]->(b:Entity)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    officer_map, entity_map,
    ('Officer', 'OFFICER_OF', 'Entity'),
    apply_friction=True
)

# 4. Intermediary INTERMEDIARY_OF Entity (1259)
load_edge(
    """
    MATCH (a:Intermediary)-[:INTERMEDIARY_OF]->(b:Entity)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    intermediary_map, entity_map,
    ('Intermediary', 'INTERMEDIARY_OF', 'Entity'),
    apply_friction=True
)

# 5. Officer BENEFICIAL_OWNER Entity (1092)
load_edge(
    """
    MATCH (a:Officer)-[:BENEFICIAL_OWNER]->(b:Entity)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    officer_map, entity_map,
    ('Officer', 'BENEFICIAL_OWNER', 'Entity'),
    apply_friction=True
)

# 6. Entity SHARES_ADDRESS Entity (308)
# No friction factor needed, same node type
load_edge(
    """
    MATCH (a:Entity)-[:SHARES_ADDRESS]->(b:Entity)
    RETURN id(a) AS source, id(b) AS target
    """,
    entity_map, entity_map,
    ('Entity', 'SHARES_ADDRESS', 'Entity'),
    apply_friction=False
)

# 7. Entity RELATED_AS Entity (128) - Identity Link
# No friction, no barrier needed given low count
load_edge(
    """
    MATCH (a:Entity)-[:RELATED_AS]->(b:Entity)
    RETURN id(a) AS source, id(b) AS target
    """,
    entity_map, entity_map,
    ('Entity', 'RELATED_AS', 'Entity'),
    apply_friction=False,
    apply_barrier=False
)

# 8. Entity CONNECTED_TO Entity (60) - Structural Shadow
load_edge(
    """
    MATCH (a:Entity)-[:CONNECTED_TO]->(b:Entity)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    entity_map, entity_map,
    ('Entity', 'CONNECTED_TO', 'Entity'),
    apply_friction=True,
    apply_barrier=False
)

# 9. Entity OFFICER_OF Intermediary (22) - Layering Signature
# No barrier needed, low count but high forensic value
load_edge(
    """
    MATCH (a:Entity)-[:OFFICER_OF]->(b:Intermediary)
    RETURN id(a) AS source, a.jurisdiction AS s_jur,
           id(b) AS target, b.jurisdiction AS t_jur
    """,
    entity_map, intermediary_map,
    ('Entity', 'OFFICER_OF', 'Intermediary'),
    apply_friction=True,
    apply_barrier=False
)

print("\nPhase 2 Complete: All nodes, features, and edges loaded.")
print(f"   Edge types loaded: {len(data.edge_types)}")
print(f"   Node types loaded: {len(data.node_types)}")

# ============================================================
# PHASE 3: HGT MODEL ARCHITECTURE
# Uses HGTConv with type-aware attention (Hu et al., 2020)
# ============================================================

class ForensicHGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        # Input linear projections per node type
        # Projects each node type to the same hidden dimension
        # regardless of their different input feature sizes
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # HGTConv layers with type-aware attention
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads
            )
            self.convs.append(conv)

        # Classification head for Entity nodes only
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to hidden dimension
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        # HGT message passing layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: F.leaky_relu(x)
                for key, x in x_dict.items()
            }

        # Classify Entity nodes only
        return self.classifier(x_dict['Entity'])


# ============================================================
# PHASE 4: GRAPH PREPARATION AND CV SETUP
# ============================================================

# Make graph undirected so all node types receive messages
print("Creating bidirectional feedback loops...")
data = T.ToUndirected()(data)

# Verify metadata
print(f"\nGraph Metadata:")
print(f"   Node types: {data.node_types}")
print(f"   Edge types: {data.edge_types}")

# Cycle entities held out separately for zero-shot inference (Phase 9)
is_test_only_tensor = torch.tensor(entity_df['is_test_only'].values, dtype=torch.long)
non_cycle_indices = torch.where(is_test_only_tensor == 0)[0]
cycle_indices = torch.where(is_test_only_tensor == 1)[0]
cycle_test_mask = cycle_indices

print(f"\nNon-cycle entities (CV pool): {len(non_cycle_indices)}")
print(f"Zero-shot cycle test set: {len(cycle_test_mask)} cycle entities (held out from training)")

# 5-fold cross validation across non-cycle entities
non_cycle_indices_np = non_cycle_indices.numpy()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for fold results
fold_metrics = []
fold_train_loss_curves = []
fold_test_loss_curves = []
fold_train_acc_curves = []
fold_test_acc_curves = []
fold_train_prec_curves = []
fold_test_prec_curves = []
all_preds_combined = []
all_targets_combined = []

# Track timing for fold 1 (representative)
HGT_HARDWARE = 'GPU' if torch.cuda.is_available() else 'CPU'
representative_train_sec = 0.0
representative_infer_sec = 0.0
representative_n_test = 0

# ============================================================
# PHASE 5: 5-FOLD CV TRAINING LOOP
# ============================================================

print(f"\n{'='*60}")
print(f"🚀 Starting 5-Fold Cross Validation for {EXPERIMENT_LABEL}")
print(f"{'='*60}")

for fold_idx, (train_split_idx, test_split_idx) in enumerate(kf.split(non_cycle_indices_np)):
    print(f"\n--- Fold {fold_idx + 1} of 5 ---")
    
    # Build masks for this fold
    train_mask = torch.tensor(non_cycle_indices_np[train_split_idx], dtype=torch.long)
    test_mask = torch.tensor(non_cycle_indices_np[test_split_idx], dtype=torch.long)
    
    print(f"  Train: {len(train_mask)} entities | Test: {len(test_mask)} entities")
    
    # Initialize fresh model for this fold
    model = ForensicHGT(
        hidden_channels=64,
        out_channels=2,
        num_heads=4,
        num_layers=3,
        metadata=data.metadata()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    pos_weight = torch.tensor([2.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Per-epoch curves for this fold
    fold_train_loss = []
    fold_test_loss = []
    fold_train_acc = []
    fold_test_acc = []
    fold_train_prec = []
    fold_test_prec = []
    
    # Time fold 1 as representative
    if fold_idx == 0:
        _train_t0 = time.perf_counter()
    
    for epoch in range(1, 201):
        # Train step
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        preds = out[train_mask, 1]
        targets = data['Entity'].y[train_mask].float()
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        train_loss_val = loss.item()
        
        # Eval step
        model.eval()
        with torch.no_grad():
            out_eval = model(data.x_dict, data.edge_index_dict)
            
            tr_pred = out_eval[train_mask].argmax(dim=-1)
            tr_targets = data['Entity'].y[train_mask]
            tr_acc = (tr_pred == tr_targets).sum().item() / len(train_mask)
            tr_tp = ((tr_pred == 1) & (tr_targets == 1)).sum().item()
            tr_fp = ((tr_pred == 1) & (tr_targets == 0)).sum().item()
            tr_prec = tr_tp / (tr_tp + tr_fp) if (tr_tp + tr_fp) > 0 else 0.0
            
            te_pred = out_eval[test_mask].argmax(dim=-1)
            te_targets = data['Entity'].y[test_mask]
            te_acc = (te_pred == te_targets).sum().item() / len(test_mask)
            te_tp = ((te_pred == 1) & (te_targets == 1)).sum().item()
            te_fp = ((te_pred == 1) & (te_targets == 0)).sum().item()
            te_prec = te_tp / (te_tp + te_fp) if (te_tp + te_fp) > 0 else 0.0
            
            te_loss = criterion(out_eval[test_mask, 1],
                                data['Entity'].y[test_mask].float()).item()
        
        fold_train_loss.append(train_loss_val)
        fold_test_loss.append(te_loss)
        fold_train_acc.append(tr_acc)
        fold_test_acc.append(te_acc)
        fold_train_prec.append(tr_prec)
        fold_test_prec.append(te_prec)
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {train_loss_val:.4f} | "
                  f"Test Acc: {te_acc:.4f} | Test Prec: {te_prec:.4f}")
    
    # Stop training timer for fold 1
    if fold_idx == 0:
        representative_train_sec = time.perf_counter() - _train_t0
        _infer_t0 = time.perf_counter()
    
    # Final evaluation for this fold
    model.eval()
    with torch.no_grad():
        out_final = model(data.x_dict, data.edge_index_dict)
        final_pred = out_final[test_mask].argmax(dim=-1)
        final_targets = data['Entity'].y[test_mask]
    
    if fold_idx == 0:
        representative_infer_sec = time.perf_counter() - _infer_t0
        representative_n_test = len(test_mask)
    
    final_pred_np = final_pred.cpu().numpy()
    final_targets_np = final_targets.cpu().numpy()
    
    acc = accuracy_score(final_targets_np, final_pred_np)
    prec = precision_score(final_targets_np, final_pred_np, zero_division=0)
    rec = recall_score(final_targets_np, final_pred_np, zero_division=0)
    f1 = f1_score(final_targets_np, final_pred_np, zero_division=0)
    
    fold_metrics.append({
        'fold': fold_idx + 1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'n_test': len(test_mask)
    })
    
    print(f"  ✅ Fold {fold_idx + 1} Final: Acc={acc:.4f} | Prec={prec:.4f} | "
          f"Rec={rec:.4f} | F1={f1:.4f}")
    
    fold_train_loss_curves.append(fold_train_loss)
    fold_test_loss_curves.append(fold_test_loss)
    fold_train_acc_curves.append(fold_train_acc)
    fold_test_acc_curves.append(fold_test_acc)
    fold_train_prec_curves.append(fold_train_prec)
    fold_test_prec_curves.append(fold_test_prec)
    
    all_preds_combined.extend(final_pred_np.tolist())
    all_targets_combined.extend(final_targets_np.tolist())

# ============================================================
# PHASE 6: AGGREGATE METRICS ACROSS FOLDS
# ============================================================

accs = [m['accuracy'] for m in fold_metrics]
precs = [m['precision'] for m in fold_metrics]
recs = [m['recall'] for m in fold_metrics]
f1s = [m['f1'] for m in fold_metrics]

mean_acc = np.mean(accs)
std_acc = np.std(accs)
mean_prec = np.mean(precs)
mean_rec = np.mean(recs)
mean_f1 = np.mean(f1s)

# Combined confusion matrix across all folds
combined_cm = confusion_matrix(all_targets_combined, all_preds_combined)
total_tn = combined_cm[0, 0]
total_fp = combined_cm[0, 1]
total_fn = combined_cm[1, 0]
total_tp = combined_cm[1, 1]

print(f"\n{'='*60}")
print(f"📊 5-Fold CV Summary: {EXPERIMENT_LABEL}")
print(f"{'='*60}")
print(f"Mean Accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Mean Precision: {mean_prec*100:.2f}%")
print(f"Mean Recall   : {mean_rec*100:.2f}%")
print(f"Mean F1       : {mean_f1*100:.2f}%")
print(f"Combined TP   : {total_tp}")
print(f"Combined FP   : {total_fp}")
print(f"Combined FN   : {total_fn}")
print(f"Combined TN   : {total_tn}")

# Append CV training timing (from fold 1 representative)
append_timing_row(
    model_name='HGT',
    train_sec=representative_train_sec,
    infer_sec=representative_infer_sec,
    n_test=representative_n_test,
    hardware=HGT_HARDWARE
)

print("\n" + "="*55)
print("⏱️  HGT CV - FOLD 1 TIMING (REPRESENTATIVE)")
print("="*55)
print(f"  Training Time      : {representative_train_sec:.4f} sec")
print(f"  Inference Time     : {representative_infer_sec:.4f} sec")
print(f"  Per-sample Inference: {(representative_infer_sec/representative_n_test)*1000:.4f} ms")
print(f"  Hardware           : {HGT_HARDWARE}")
print("="*55 + "\n")

# ============================================================
# PHASE 7: VISUALIZATIONS FROM CV
# ============================================================

# Combined confusion matrix figure
print("\n Generating combined confusion matrix from CV...")
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=combined_cm,
    display_labels=['Legitimate (UK)', 'Shell (ICIJ)']
)
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title(f'Confusion Matrix - {EXPERIMENT_LABEL}\nCombined Across 5 Folds', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{EXPERIMENT}_confusion_matrix.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f" Confusion matrix saved to: {OUTPUT_DIR}")

# Precision convergence curves (overlay 5 folds)
print("\n Generating CV precision convergence plot...")
plt.figure(figsize=(10, 6))
epochs_range = range(1, 201)
colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
for fold_idx in range(5):
    plt.plot(epochs_range, fold_train_prec_curves[fold_idx],
             color=colors[fold_idx], alpha=0.4, linewidth=1,
             linestyle='--')
    plt.plot(epochs_range, fold_test_prec_curves[fold_idx],
             color=colors[fold_idx], alpha=0.85, linewidth=1.5,
             label=f'Fold {fold_idx + 1} Test')
plt.title(f'{EXPERIMENT_LABEL} - Precision Convergence Across 5 Folds\n'
          f'(Solid = Test, Dashed = Train)', fontsize=11)
plt.xlabel('Epoch')
plt.ylabel('Precision Score')
plt.ylim(0.0, 1.05)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{EXPERIMENT}_precision_curve.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f" Precision curve saved to: {OUTPUT_DIR}")

# ============================================================
# PHASE 8: RESULTS SUMMARY CSV
# ============================================================
results_summary = pd.DataFrame([{
    'Model': EXPERIMENT_LABEL,
    'CV_Folds': 5,
    'Total_Entities': int(len(non_cycle_indices)),
    'Mean_Accuracy': round(mean_acc, 4),
    'Std_Accuracy': round(std_acc, 4),
    'Mean_Precision': round(mean_prec, 4),
    'Mean_Recall': round(mean_rec, 4),
    'Mean_F1': round(mean_f1, 4),
    'Combined_TP': int(total_tp),
    'Combined_FP': int(total_fp),
    'Combined_FN': int(total_fn),
    'Combined_TN': int(total_tn),
    'Epochs': 200,
    'HGT_Layers': 3,
    'Attention_Heads': 4,
    'Hidden_Channels': 64,
    'Learning_Rate': 0.001,
    'Weight_Decay': 1e-4,
    'Pos_Weight': 2.0,
    'K_Barrier': 25,
    'Friction_Factor': 1.25,
    'Random_Seed': 42
}])
results_summary.to_csv(
    os.path.join(OUTPUT_DIR, f'{EXPERIMENT}_results_summary.csv'),
    index=False
)
print(f" Results summary saved to: {OUTPUT_DIR}")

# ============================================================
# PHASE 9: FINAL TRAINING PASS FOR ZERO-SHOT INFERENCE
# Trains a final canonical model on the entire non-cycle pool
# This model is used for cycle inference and t-SNE visualization
# ============================================================
print("\n" + "="*65)
print(" PHASE 9: Final Training Pass on Full Non-Cycle Pool")
print("="*65)
print(f"   Training on {len(non_cycle_indices)} non-cycle entities")
print(f"   Cycle entities ({len(cycle_test_mask)}) held out for zero-shot test")

# Initialize fresh canonical model
model = ForensicHGT(
    hidden_channels=64,
    out_channels=2,
    num_heads=4,
    num_layers=3,
    metadata=data.metadata()
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
pos_weight = torch.tensor([2.0])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Train on full non-cycle pool
final_train_mask = non_cycle_indices

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    preds = out[final_train_mask, 1]
    targets = data['Entity'].y[final_train_mask].float()
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f}")

print(" Final canonical model trained.")

# Save canonical weights
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{EXPERIMENT}_hgt_weights.pth'))
print(f" Canonical model weights saved to {OUTPUT_DIR}")

# T-SNE Latent Space using canonical model
print("\n Generating t-SNE Latent Space from Canonical Model...")
model.eval()
with torch.no_grad():
    x_dict = {
        node_type: model.lin_dict[node_type](x).relu()
        for node_type, x in data.x_dict.items()
    }
    for conv in model.convs:
        x_dict = conv(x_dict, data.edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

    embeddings = x_dict['Entity'].cpu().numpy()
    labels = data['Entity'].y.cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    emb_2d[:, 0], emb_2d[:, 1],
    c=labels, cmap='coolwarm',
    alpha=0.7, edgecolors='w', linewidths=0.5
)
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=['Legitimate (UK)', 'Shell (ICIJ)']
)
plt.title(
    f'{EXPERIMENT_LABEL} — Latent Space (Canonical Model)\nType-Aware Heterogeneous Graph Transformer'
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, f'{EXPERIMENT}_hgt_tsne.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f" t-SNE saved to: {OUTPUT_DIR}")

# ============================================================
# PHASE 10: ZERO-SHOT INFERENCE ON ICIJ CIRCULAR OWNERSHIP TEST SET
# Uses the canonical model from Phase 9
# ============================================================
print("\n Phase 10: Zero-Shot Inference on ICIJ Circular Ownership Test Set")
print("="*65)

if len(cycle_test_mask) == 0:
    print(" No cycle test entities found in graph (is_test_only=1). Skipping Phase 10.")
else:
    # === COMP ANALYSIS: cycle inference timer start ===
    _cycle_infer_t0 = time.perf_counter()
    # ===================================================

    model.eval()
    with torch.no_grad():
        out_full = model(data.x_dict, data.edge_index_dict)
        cycle_preds = out_full[cycle_test_mask].argmax(dim=-1)
        cycle_targets = data['Entity'].y[cycle_test_mask]

    # === COMP ANALYSIS: cycle inference timer end ===
    _cycle_infer_elapsed = time.perf_counter() - _cycle_infer_t0
    print(f"⏱  Zero-shot inference wall-clock: {_cycle_infer_elapsed:.4f}s on {HGT_HARDWARE}")
    # =================================================

    # Compute confusion matrix
    cycle_tp = ((cycle_preds == 1) & (cycle_targets == 1)).sum().item()
    cycle_fp = ((cycle_preds == 1) & (cycle_targets == 0)).sum().item()
    cycle_fn = ((cycle_preds == 0) & (cycle_targets == 1)).sum().item()
    cycle_tn = ((cycle_preds == 0) & (cycle_targets == 0)).sum().item()

    cycle_precision = cycle_tp / (cycle_tp + cycle_fp) if (cycle_tp + cycle_fp) > 0 else 0.0
    cycle_recall = cycle_tp / (cycle_tp + cycle_fn) if (cycle_tp + cycle_fn) > 0 else 0.0
    cycle_f1 = 2 * (cycle_precision * cycle_recall) / (cycle_precision + cycle_recall) \
               if (cycle_precision + cycle_recall) > 0 else 0.0
    cycle_accuracy = (cycle_tp + cycle_tn) / (cycle_tp + cycle_tn + cycle_fp + cycle_fn)

    print("\n--- ZERO-SHOT FORENSIC PERFORMANCE (CIRCULAR OWNERSHIP TEST SET) ---")
    print(f"Total Cycle Test Entities : {len(cycle_test_mask)}")
    print(f"True Shells   (TP)        : {cycle_tp}")
    print(f"Missed Shells (FN)        : {cycle_fn}")
    print(f"False Alarms  (FP)        : {cycle_fp}")
    print(f"Correct Legit (TN)        : {cycle_tn}")
    print("-" * 60)
    print(f"Accuracy  : {cycle_accuracy:.4f}")
    print(f"Precision : {cycle_precision:.4f}")
    print(f"Recall    : {cycle_recall:.4f}")
    print(f"F1-Score  : {cycle_f1:.4f}")

    # ============================================================
    # PHASE 10.1: PER-ENTITY PREDICTION TABLE
    # ============================================================
    print("\n Generating Per-Entity Prediction Table...")

    cycle_metadata_query = """
        MATCH (e:Entity {is_test_only: 1})
        RETURN id(e) AS node_id, e.name AS name, e.jurisdiction AS jurisdiction,
               e.in_degree AS in_degree, e.out_degree AS out_degree
    """
    cycle_meta_df = gds.run_cypher(cycle_metadata_query)

    pyg_to_neo = {v: k for k, v in entity_map.items()}
    cycle_neo_ids = [pyg_to_neo[idx.item()] for idx in cycle_test_mask]

    cycle_results_df = pd.DataFrame({
        'neo4j_node_id': cycle_neo_ids,
        'predicted_label': cycle_preds.cpu().numpy(),
        'true_label': cycle_targets.cpu().numpy(),
        'correct': (cycle_preds == cycle_targets).cpu().numpy()
    })
    cycle_results_df = cycle_results_df.merge(
        cycle_meta_df, left_on='neo4j_node_id', right_on='node_id', how='left'
    ).drop(columns=['node_id'])

    cycle_results_df['outcome'] = cycle_results_df['correct'].apply(
        lambda c: ' Detected' if c else '❌ Missed'
    )

    print(cycle_results_df[['name', 'jurisdiction', 'in_degree', 'out_degree',
                             'predicted_label', 'true_label', 'outcome']].to_string(index=False))

    cycle_results_df.to_csv(
        os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_cycle_per_entity.csv'),
        index=False
    )
    print(f" Per-entity predictions saved to: {CYCLE_OUTPUT_DIR}")

    # ============================================================
    # PHASE 10.2: STRUCTURAL BREAKDOWN
    # ============================================================
    print("\n Structural Breakdown of Cycle Predictions...")

    by_jur = cycle_results_df.groupby('jurisdiction').agg(
        n_total=('correct', 'count'),
        n_detected=('correct', 'sum')
    ).reset_index()
    by_jur['detection_rate'] = by_jur['n_detected'] / by_jur['n_total']
    print("\nBy jurisdiction:")
    print(by_jur.to_string(index=False))

    cycle_results_df['degree_bucket'] = pd.cut(
        cycle_results_df['in_degree'],
        bins=[0, 2, 5, 100],
        labels=['low (1-2)', 'mid (3-5)', 'high (6+)']
    )
    by_deg = cycle_results_df.groupby('degree_bucket', observed=True).agg(
        n_total=('correct', 'count'),
        n_detected=('correct', 'sum')
    ).reset_index()
    by_deg['detection_rate'] = by_deg['n_detected'] / by_deg['n_total']
    print("\nBy degree bucket:")
    print(by_deg.to_string(index=False))

    by_jur.to_csv(os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_cycle_by_jurisdiction.csv'), index=False)
    by_deg.to_csv(os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_cycle_by_degree.csv'), index=False)

    # ============================================================
    # PHASE 10.3: T-SNE WITH CYCLE ENTITIES HIGHLIGHTED
    # ============================================================
    print("\n Generating t-SNE with Cycle Entities Highlighted...")

    model.eval()
    with torch.no_grad():
        x_dict = {
            node_type: model.lin_dict[node_type](x).relu()
            for node_type, x in data.x_dict.items()
        }
        for conv in model.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        embeddings_full = x_dict['Entity'].cpu().numpy()
        labels_full = data['Entity'].y.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d_full = tsne.fit_transform(embeddings_full)

    plot_class = labels_full.copy()
    cycle_idx_np = cycle_test_mask.cpu().numpy()
    plot_class[cycle_idx_np] = 2

    plt.figure(figsize=(10, 8))
    legit_mask = plot_class == 0
    shell_mask = plot_class == 1
    cycle_mask = plot_class == 2

    plt.scatter(emb_2d_full[legit_mask, 0], emb_2d_full[legit_mask, 1],
                c='#1F77B4', alpha=0.5, s=20, label='Legitimate (UK)')
    plt.scatter(emb_2d_full[shell_mask, 0], emb_2d_full[shell_mask, 1],
                c='#D62728', alpha=0.5, s=20, label='Shell (ICIJ Training)')
    plt.scatter(emb_2d_full[cycle_mask, 0], emb_2d_full[cycle_mask, 1],
                c='#2CA02C', alpha=0.95, s=120, marker='*',
                edgecolors='black', linewidths=1.0,
                label='Circular Ownership (Zero-Shot Test)')
    plt.legend(loc='best')
    plt.title(
        f'{EXPERIMENT_LABEL} — Latent Space with Zero-Shot Cycle Entities\n'
        f'Type-Aware Heterogeneous Graph Transformer'
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(
        os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_tsne_with_cycles.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f" t-SNE with cycles saved to: {CYCLE_OUTPUT_DIR}")

    # ============================================================
    # PHASE 10.4: MAIN TEST VS CYCLE TEST COMPARISON
    # ============================================================
    print("\n Generating Main Test vs Cycle Test Comparison Bar Chart...")

    metrics_compare = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Main Test Set (5-Fold CV Mean)': [mean_acc, mean_prec, mean_rec, mean_f1],
        'Cycle Test Set': [cycle_accuracy, cycle_precision, cycle_recall, cycle_f1]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(metrics_compare))
    width = 0.35
    ax.bar([p - width/2 for p in x_pos], metrics_compare['Main Test Set (5-Fold CV Mean)'],
           width=width, label='Main Test Set (5-Fold CV Mean)', color='#1F77B4')
    ax.bar([p + width/2 for p in x_pos], metrics_compare['Cycle Test Set'],
           width=width, label='Cycle Test Set (Zero-Shot)', color='#2CA02C')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(metrics_compare['Metric'])
    ax.set_ylabel('Score')
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f'{EXPERIMENT_LABEL} — Main Test (CV) vs Zero-Shot Cycle Test\n'
        f'Generalisation to Circular Ownership Patterns'
    )
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for i, (m, c) in enumerate(zip(metrics_compare['Main Test Set (5-Fold CV Mean)'],
                                     metrics_compare['Cycle Test Set'])):
        ax.text(i - width/2, m + 0.01, f'{m:.3f}', ha='center', fontsize=9)
        ax.text(i + width/2, c + 0.01, f'{c:.3f}', ha='center', fontsize=9)
    plt.savefig(
        os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_main_vs_cycle_comparison.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f" Comparison chart saved to: {CYCLE_OUTPUT_DIR}")

    # ============================================================
    # PHASE 10.5: CYCLE TEST CONFUSION MATRIX HEATMAP
    # ============================================================
    print("\n Generating Cycle Test Confusion Matrix Heatmap...")

    cycle_cm = pd.DataFrame(
        [[cycle_tn, cycle_fp], [cycle_fn, cycle_tp]],
        index=['Actual Legit', 'Actual Shell'],
        columns=['Predicted Legit', 'Predicted Shell']
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cycle_cm.values, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cycle_cm.columns)
    ax.set_yticklabels(cycle_cm.index)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cycle_cm.values[i, j]),
                    ha='center', va='center', fontsize=20,
                    color='white' if cycle_cm.values[i, j] > cycle_cm.values.max()/2 else 'black')
    ax.set_title(
        f'{EXPERIMENT_LABEL} — Cycle Test Confusion Matrix\n'
        f'Zero-Shot Inference on {len(cycle_test_mask)} Circular Ownership Entities'
    )
    plt.colorbar(im, ax=ax)
    plt.savefig(
        os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_cycle_confusion_matrix.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f" Confusion matrix saved to: {CYCLE_OUTPUT_DIR}")

    # ============================================================
    # PHASE 10.6: CYCLE RESULTS SUMMARY
    # ============================================================
    cycle_summary = pd.DataFrame([{
        'Model': EXPERIMENT_LABEL + ' (Zero-Shot Cycle Test)',
        'Test_Entities': len(cycle_test_mask),
        'Accuracy': round(cycle_accuracy, 4),
        'Precision': round(cycle_precision, 4),
        'Recall': round(cycle_recall, 4),
        'F1_Score': round(cycle_f1, 4),
        'True_Positives': cycle_tp,
        'False_Positives': cycle_fp,
        'False_Negatives': cycle_fn,
        'True_Negatives': cycle_tn,
        'Inference_Time_Sec': round(_cycle_infer_elapsed, 4),
        'Per_Sample_Inference_Ms': round(_cycle_infer_elapsed / len(cycle_test_mask) * 1000, 4),
        'Hardware': HGT_HARDWARE
    }])
    cycle_summary.to_csv(
        os.path.join(CYCLE_OUTPUT_DIR, f'{EXPERIMENT}_cycle_results_summary.csv'),
        index=False
    )

    append_timing_row(
        model_name='HGT_Cycle_ZeroShot',
        train_sec=0.0,
        infer_sec=_cycle_infer_elapsed,
        n_test=len(cycle_test_mask),
        hardware=HGT_HARDWARE
    )

    print("\n" + "="*65)
    print(f" Phase 10 Complete: Zero-Shot Cycle Inference Results Saved to {CYCLE_OUTPUT_DIR}")
    print("="*65)