import torch
import pandas as pd
from torch_geometric.data import HeteroData
from graphdatascience import GraphDataScience

#....PHASE 2: DATA ENGINE....

# 1. Handshake with Neo4j
gds = GraphDataScience("neo4j://127.0.0.1:7687", auth=("neo4j", "Vladivostok1."))

# 2. Load the External Authority (The CSV you created)
# Ensure 'jurisdiction_risk.csv' is in the same folder as this script
risk_lookup = pd.read_csv('jurisdiction_risk.csv')

data = HeteroData()

# 3. Pull Entity Nodes & Map 3D Features
entity_query = """
    MATCH (e:Entity) 
    RETURN id(e) as node_id, e.is_shell as label, e.jurisdiction as jurisdiction
"""
entity_df = gds.run_cypher(entity_query)

# Join with CSV to get f1, f2, f3
entity_df = entity_df.merge(risk_lookup, on='jurisdiction', how='left').fillna(0)

# Load into PyG
data['Entity'].x = torch.tensor(entity_df[['f1_secrecy', 'f2_facilitation', 'f3_governance']].values, dtype=torch.float)
data['Entity'].y = torch.tensor(entity_df['label'].values, dtype=torch.long)

# 4. Pull Officer Nodes (Repeat the same logic)
officer_query = "MATCH (o:Officer) RETURN id(o) as node_id, o.jurisdiction as jurisdiction"
officer_df = gds.run_cypher(officer_query)
officer_df = officer_df.merge(risk_lookup, on='jurisdiction', how='left').fillna(0)
data['Officer'].x = torch.tensor(officer_df[['f1_secrecy', 'f2_facilitation', 'f3_governance']].values, dtype=torch.float)

# 5. The Friction Factor (phi) Calculation
# Pull edges and jurisdictions together
bo_query = """
    MATCH (o:Officer)-[:BENEFICIAL_OWNER]->(e:Entity)
    RETURN id(o) as source, o.jurisdiction as s_jur, id(e) as target, e.jurisdiction as t_jur
"""
bo_df = gds.run_cypher(bo_query)

# Create ID mappings (Neo4j ID -> local 0,1,2...)
entity_map = {old: i for i, old in enumerate(entity_df['node_id'])}
officer_map = {old: i for i, old in enumerate(officer_df['node_id'])}

# Translate IDs and Calculate phi
src_idx = [officer_map[s] for s in bo_df['source']]
dst_idx = [entity_map[d] for d in bo_df['target']]
weights = [1.25 if row['s_jur'] != row['t_jur'] else 1.0 for _, row in bo_df.iterrows()]

data['Officer', 'BENEFICIAL_OWNER', 'Entity'].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
data['Officer', 'BENEFICIAL_OWNER', 'Entity'].edge_weight = torch.tensor(weights, dtype=torch.float)

print("✅ Phase 2 Complete: Nodes and Friction Factors are loaded in VS Code.")


#---------PHASE 3: GNN MODELING (Outline)---------
import torch.nn.functional as F
import torch_geometric.transforms as T
# 1. Added GraphConv to the imports
from torch_geometric.nn import GraphConv, HeteroConv 

class ForensicGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            # 2. Swapped SAGEConv for GraphConv to support edge_weight
            conv = HeteroConv({
                edge_type: GraphConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='add')
            self.convs.append(conv)

        self.lin1 = torch.nn.Linear(3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        # --- ABLATION STUDY: THE LOBOTOMY ---
        # We completely comment out the message passing loop.
        # The model no longer looks at neighbors, edges, or Friction Factors.
        
        # for conv in self.convs:
        #     x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        #     x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        #     x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # We extract the raw Entity features directly, bypassing the graph structure
        out = x_dict['Entity']
        out = self.lin1(out).relu()
        return self.lin2(out)


# --- THE FORENSIC IGNITION SEQUENCE ---

# 1. Force the graph to be bidirectional so Officer nodes don't disappear
print("🔄 Creating bidirectional feedback loops...")
data = T.ToUndirected()(data)

# 2. Dynamically map ALL edge weights (including the newly created reverse edges)
data.edge_weight_dict = {
    edge_type: data[edge_type].edge_weight for edge_type in data.edge_types
}

# 3. Initialize Model & Optimizer
model = ForensicGNN(hidden_channels=64, out_channels=2, metadata=data.metadata())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Weighted BCE Loss & Masks
pos_weight = torch.tensor([2.0]) 
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

num_entities = data['Entity'].num_nodes
indices = torch.randperm(num_entities)
train_mask = indices[:int(num_entities * 0.8)]
test_mask = indices[int(num_entities * 0.8):]

def train():
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    out = model(data.x_dict, data.edge_index_dict, data.edge_weight_dict)
    
    # 'out' is a 2D tensor of shape [N, 2]. 
    # We cleanly slice the training rows and the 'Shell' prediction column (index 1)
    preds = out[train_mask, 1]
    targets = data['Entity'].y[train_mask].float()
    
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, data.edge_weight_dict)
        # Check which logic (0 or 1) has the higher probability
        pred = out[mask].argmax(dim=-1)
        correct = (pred == data['Entity'].y[mask]).sum().item()
        return correct / len(mask)

# 5. The Training Loop
print("🚀 Starting Training Phase...")
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc = evaluate(train_mask)
        test_acc = evaluate(test_mask)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print("✅ Training Complete. The Forensic Engine is now active.")




# --- THE FORENSIC AUDIT (PURE PYTORCH METRICS) ---
print("\n🔍 Running Final Forensic Audit on Unseen Test Data...")

model.eval()
with torch.no_grad():
    # Run the forward pass
    out = model(data.x_dict, data.edge_index_dict, data.edge_weight_dict)
    
    # Get predictions and ground truth for the TEST mask
    preds = out[test_mask].argmax(dim=-1)
    targets = data['Entity'].y[test_mask]

# Calculate True Positives, False Positives, False Negatives, True Negatives
# "Positive" (1) = Shell Company (ICIJ)
# "Negative" (0) = Legitimate Company (UK)
tp = ((preds == 1) & (targets == 1)).sum().item()
fp = ((preds == 1) & (targets == 0)).sum().item()
fn = ((preds == 0) & (targets == 1)).sum().item()
tn = ((preds == 0) & (targets == 0)).sum().item()

# Calculate Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n--- FORENSIC PERFORMANCE (TEST SET) ---")
print(f"Total Test Entities: {len(test_mask)}")
print(f"True Shells Found (TP): {tp}")
print(f"Missed Shells (FN): {fn}")
print(f"False Alarms (FP): {fp}")
print(f"Correct Legitimate (TN): {tn}")
print("-" * 35)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (When it flags a shell, how often is it right?)")
print(f"Recall:    {recall:.4f} (Out of all real shells, how many did it catch?)")
print(f"F1-Score:  {f1_score:.4f} (The ultimate forensic benchmark)")

torch.save(model.state_dict(), 'vibrant_gnn_forensic_weights.pth')
print("💾 Model weights successfully saved to disk.")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- T-SNE VISUALIZATION: THE LOBOTOMIZED MODEL ---
print("\n🎨 Generating Ablation t-SNE Visualization...")

model.eval()
with torch.no_grad():
    # In the ablation model, we bypassed the graph. 
    # We grab the raw 3D features and pass them through the first linear layer.
    raw_features = data.x_dict['Entity']
    embeddings = model.lin1(raw_features).relu().cpu().numpy()
    labels = data['Entity'].y.cpu().numpy()

# Compress 64 dimensions into 2
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# Plot the scatter map
plt.figure(figsize=(10, 8))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7, edgecolors='w', linewidths=0.5)
plt.legend(handles=scatter.legend_elements()[0], labels=['Legitimate (UKCH)', 'Shell (ICIJ)'], loc='best')
plt.title('Ablation Model Latent Space (Flat Features Only)\nNotice the heavy overlap and lack of structure.')
plt.grid(True, linestyle='--', alpha=0.5)

# Save it to your folder
plt.savefig('GNN_ablation.png', dpi=300, bbox_inches='tight')
print("✅ Ablation plot successfully saved as 'GNN_ablation.png'.")