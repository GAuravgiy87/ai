import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

# Set output directory
output_dir = r"c:\Users\USER\Desktop\ai\docs"
os.makedirs(output_dir, exist_ok=True)

# Use basic font if Times New Roman is not installed properly, to avoid warnings
plt.rcParams['font.family'] = 'serif'

# ---------------------------------------------------------
# 1. Bar Chart
# ---------------------------------------------------------
labels = ['S1: Office Corridor', 'S2: Lobby', 'S4: Low Light', 'S5: Crowded']
precision = [94.2, 91.8, 82.3, 79.6]
recall = [91.7, 89.2, 78.9, 74.3]
map05 = [93.1, 90.4, 80.6, 76.9]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
rects2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
rects3 = ax.bar(x + width, map05, width, label='mAP@0.5', color='#2ca02c')

ax.set_ylabel('Performance (%)', fontsize=12)
ax.set_title('Detection Performance Across Scenarios', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.set_ylim(60, 100)
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_yticks(np.arange(60, 101, 5))
ax.legend(prop={'size': 10}, loc='upper right')

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_graph.png'), dpi=300)
plt.close(fig)

# ---------------------------------------------------------
# 2. Pipeline Flowchart
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10), dpi=600)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

steps = [
    ("Step 1 — Data Collection", "RTSP stream ingestion via threading queue"),
    ("Step 2 — Preprocessing", "Gamma correction + CLAHE + Saturation boost via OpenCL UMat"),
    ("Step 3 — Detection & Recognition", "YOLOv8s person detection → MTCNN alignment → FaceNet embedding"),
    ("Step 4 — Analysis", "Hungarian algorithm tracker + HSV appearance model association"),
    ("Step 5 — Output Generation", "Frame rendering → JPEG encoding → Redis dispatch"),
    ("Step 6 — Dashboard & Storage", "SSE analytics stream + FFmpeg MKV recording")
]

y_pos = 12
for i, (title, desc) in enumerate(steps):
    # draw box
    box = FancyBboxPatch((1.5, y_pos), 7, 1.2, boxstyle="round,pad=0.2", ec="#1e4d8a", fc="#e6f2ff", lw=1.5)
    ax.add_patch(box)
    ax.text(5, y_pos + 0.8, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#003366')
    ax.text(5, y_pos + 0.4, desc, ha='center', va='center', fontsize=10, wrap=True)
    
    if i < len(steps) - 1:
        # draw arrow
        ax.annotate('', xy=(5, y_pos - 0.7), xytext=(5, y_pos),
                    arrowprops=dict(facecolor='#1e4d8a', shrink=0, width=2, headwidth=8, edgecolor='#1e4d8a'))
    y_pos -= 2

plt.savefig(os.path.join(output_dir, 'pipeline_workflow.png'), dpi=600, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------
# 3. System Architecture Diagram
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(x, y, w, h, title, items, color, border_color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", ec=border_color, fc=color, lw=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.5, title, ha='center', va='center', fontsize=11, fontweight='bold')
    y_item = y + h - 1.2
    for item in items:
        ax.text(x + 0.3, y_item, f"• {item}", ha='left', va='center', fontsize=9.5)
        y_item -= 0.6

# Input layer (blue)
draw_box(0.5, 3.5, 3, 3, "INPUT LAYER", 
         ["Multiple IP Cameras\n  (RTSP streams)", "Threading Queue\n  (buffer management)"], 
         "#e6f0fa", "#2a5b84")

# Processing layer (green)
draw_box(4.2, 2.5, 3.8, 5, "PROCESSING LAYER\n(Camera Server - Port 9001)", 
         ["YOLOv8s Object Detection", "MTCNN Face Alignment", "FaceNet Embedding Extraction", "Hungarian Algorithm Tracker\n  (with HSV Appearance Model)", "Adaptive Resource Guard (ARG)"], 
         "#e6fae6", "#2a842a")

# Data layer (orange)
draw_box(8.7, 5.5, 3, 3, "DATA LAYER", 
         ["PostgreSQL Database\n  (identity gallery, logs)", "Redis Stream\n  (frame transport)"], 
         "#faeee6", "#844b2a")

# Output layer (purple)
draw_box(8.7, 1.5, 3, 3, "OUTPUT LAYER", 
         ["Main App Dashboard\n  (Port 9000)", "FFmpeg Recording Worker\n  (MKV crash-safe files)", "SSE Analytics Stream"], 
         "#f2e6fa", "#5b2a84")

# Arrows
# Input to Processing
ax.annotate('', xy=(4.2, 5), xytext=(3.5, 5), arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=7))
# Processing to Data
ax.annotate('', xy=(8.7, 7), xytext=(8.0, 7), arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=7))
# Processing to Output
ax.annotate('', xy=(8.7, 3), xytext=(8.0, 3), arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=7))

plt.savefig(os.path.join(output_dir, 'system_architecture.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print("Images generated successfully.")
