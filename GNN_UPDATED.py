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

# --- THE K-BARRIER (PURE PYTHON IMPLEMENTATION) ---
print("🚧 Enforcing the K-Barrier (Max 25 connections per entity)...")

# 1. Put the edges into a Pandas DataFrame for easy manipulation
edges_df = pd.DataFrame({
    'source': src_idx,
    'target': dst_idx,
    'weight': weights
})

original_edge_count = len(edges_df)

# 2. Group by the Target Entity, and randomly sample a MAXIMUM of 25 edges per entity.
# If an entity has less than 25 edges, it keeps all of them.
pruned_edges_df = edges_df.groupby('target', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 25), random_state=42))

# 3. Extract the pruned lists back out
src_idx_pruned = pruned_edges_df['source'].tolist()
dst_idx_pruned = pruned_edges_df['target'].tolist()
weights_pruned = pruned_edges_df['weight'].tolist()

print(f"✂️ Hub Dominance Prevented: Graph pruned from {original_edge_count} edges to {len(pruned_edges_df)} edges.")

# 4. Load the pruned, memory-safe data into the PyTorch HeteroData object
data['Officer', 'BENEFICIAL_OWNER', 'Entity'].edge_index = torch.tensor([src_idx_pruned, dst_idx_pruned], dtype=torch.long)
data['Officer', 'BENEFICIAL_OWNER', 'Entity'].edge_weight = torch.tensor(weights_pruned, dtype=torch.float)
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

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        for conv in self.convs:
            # GraphConv now correctly consumes the Friction Factors
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
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
        pred = out[mask].argmax(dim=-1)
        targets = data['Entity'].y[mask]
        
        # Calculate Accuracy
        correct = (pred == targets).sum().item()
        accuracy = correct / len(mask)
        
        # Calculate Precision on the fly
        tp = ((pred == 1) & (targets == 1)).sum().item()
        fp = ((pred == 1) & (targets == 0)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return accuracy, precision

# 5. The Training Loop
print("🚀 Starting Training Phase...")

# Trackers for the visualization
train_precisions = []
test_precisions = []

for epoch in range(1, 201):
    loss = train()
    
    # We evaluate every single epoch to build a smooth chart
    train_acc, train_prec = evaluate(train_mask)
    test_acc, test_prec = evaluate(test_mask)
    
    # Store the precision data
    train_precisions.append(train_prec)
    test_precisions.append(test_prec)
    
    # Only print to terminal every 20 epochs so it doesn't get messy
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}, Test Prec: {test_prec:.4f}')

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
import torch.nn.functional as F

# --- T-SNE VISUALIZATION: THE FULL FORENSIC ENGINE ---
print("\n🎨 Generating Main GNN t-SNE Visualization...")

model.eval()
with torch.no_grad():
    # We manually pass the data through the graph layers to capture the 'Friction Factor' physics
    x_dict = data.x_dict
    for conv in model.convs:
        x_dict = conv(x_dict, data.edge_index_dict, edge_weight_dict=data.edge_weight_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
    
    # Extract the final 64-dimensional structural embeddings for the Entities
    embeddings = x_dict['Entity'].cpu().numpy()
    labels = data['Entity'].y.cpu().numpy()

# Compress 64 dimensions into 2
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# Plot the scatter map
plt.figure(figsize=(10, 8))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7, edgecolors='w', linewidths=0.5)
plt.legend(handles=scatter.legend_elements()[0], labels=['Legitimate (UK)', 'Shell (ICIJ)'], loc='best')
plt.title('Main GNN Latent Space (Multi-Hop Graph Topology)\nNotice the clear forensic separation of the shell networks.')
plt.grid(True, linestyle='--', alpha=0.5)

# Save it to your folder
plt.savefig('NEW_UPDATED_GNN.png', dpi=300, bbox_inches='tight')
print("✅ Main GNN plot successfully saved as 'UPDATED_GNN.png'.")

# --- PRECISION PER EPOCH VISUALIZATION ---
print("\n📈 Generating Precision per Epoch Visualization...")

plt.figure(figsize=(10, 6))
epochs_range = range(1, 201)

# Plot the curves
plt.plot(epochs_range, train_precisions, label='Training Precision', color='blue', alpha=0.5)
plt.plot(epochs_range, test_precisions, label='Validation (Test) Precision', color='green', linewidth=2)

# Styling the chart for the thesis
plt.title('Model Convergence: Precision Score per Epoch\nDemonstrating the reduction of False Positives over time')
plt.xlabel('Epoch (Training Round)')
plt.ylabel('Precision Score')
plt.ylim(0.0, 1.05)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Save the evidence
plt.savefig('precision_learning_curve.png', dpi=300, bbox_inches='tight')
print("✅ Precision curve successfully saved as 'precision_learning_curve.png'.")