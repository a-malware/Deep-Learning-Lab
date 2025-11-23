#!/usr/bin/env python3
"""
Generate system architecture diagram for the report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
color_input = '#FFB366'
color_face = '#66B3FF'
color_gesture = '#99FF99'
color_association = '#FF99CC'
color_output = '#FFB366'

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10, style='round'):
    if style == 'round':
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
    else:
        box = Rectangle((x-width/2, y-height/2), width, height,
                       edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            weight='bold', multialignment='center')

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# Input
create_box(ax, 5, 11, 2, 0.6, 'Webcam Input\n640×480', color_input, fontsize=11)

# Face Recognition Module (Left side)
# Module box
face_module = FancyBboxPatch((0.5, 5.5), 3.5, 4,
                            boxstyle="round,pad=0.15", 
                            edgecolor='blue', facecolor='none', 
                            linewidth=2, linestyle='--')
ax.add_patch(face_module)
ax.text(2.25, 9.7, 'Face Recognition Module', ha='center', va='top', 
        fontsize=11, weight='bold', color='blue')

create_box(ax, 2.25, 8.8, 2.5, 0.7, 'Face Detection\nHaar Cascade', color_face, fontsize=9)
create_box(ax, 2.25, 7.5, 2.5, 0.7, 'Face Recognition\nLBPH Algorithm', color_face, fontsize=9)
create_box(ax, 2.25, 6.2, 2.2, 0.7, 'Trained Model\n9 Individuals', color_face, fontsize=9)

# Gesture Recognition Module (Right side)
# Module box
gesture_module = FancyBboxPatch((6, 5.5), 3.5, 4,
                               boxstyle="round,pad=0.15", 
                               edgecolor='green', facecolor='none', 
                               linewidth=2, linestyle='--')
ax.add_patch(gesture_module)
ax.text(7.75, 9.7, 'Gesture Recognition Module', ha='center', va='top', 
        fontsize=11, weight='bold', color='green')

create_box(ax, 7.75, 8.8, 2.5, 0.7, 'Hand Detection\nONNX Model', color_gesture, fontsize=9)
create_box(ax, 7.75, 7.5, 2.5, 0.7, 'Gesture Classification\nONNX Model', color_gesture, fontsize=9)
create_box(ax, 7.75, 6.2, 2.5, 0.7, 'Hand Tracking\nOC-SORT', color_gesture, fontsize=9)

# Association Module
create_box(ax, 5, 4.2, 3.5, 0.8, 'Spatial Association\nDistance-based Matching', 
          color_association, fontsize=10)

# Output
create_box(ax, 5, 2.5, 2.5, 0.7, 'Output\nPerson + Gesture', color_output, fontsize=11)

# Visualization
create_box(ax, 5, 1, 2.5, 0.6, 'Real-time Display\nConsole Output', color_output, fontsize=10)

# Arrows - Input to modules
create_arrow(ax, 4.2, 10.7, 2.8, 9.2)
create_arrow(ax, 5.8, 10.7, 7.2, 9.2)

# Arrows - Within face module
create_arrow(ax, 2.25, 8.45, 2.25, 7.85)
create_arrow(ax, 2.25, 7.15, 2.25, 6.55)

# Arrows - Within gesture module
create_arrow(ax, 7.75, 8.45, 7.75, 7.85)
create_arrow(ax, 7.75, 7.15, 7.75, 6.55)

# Arrows - Modules to association
create_arrow(ax, 2.25, 5.85, 3.5, 4.6)
create_arrow(ax, 7.75, 5.85, 6.5, 4.6)

# Arrow - Association to output
create_arrow(ax, 5, 3.8, 5, 2.85)

# Arrow - Output to display
create_arrow(ax, 5, 2.15, 5, 1.3)

# Add labels on arrows
ax.text(3.5, 9.8, 'Frame', fontsize=8, style='italic')
ax.text(6.5, 9.8, 'Frame', fontsize=8, style='italic')
ax.text(1.5, 8.15, 'Face\nRegions', fontsize=7, style='italic', ha='center')
ax.text(1.5, 6.85, 'Face\nIDs', fontsize=7, style='italic', ha='center')
ax.text(8.8, 8.15, 'Hand\nBBoxes', fontsize=7, style='italic', ha='center')
ax.text(8.8, 6.85, 'Gesture\nLabels', fontsize=7, style='italic', ha='center')
ax.text(2.8, 5.2, 'Faces +\nIDs', fontsize=7, style='italic', ha='center')
ax.text(7.2, 5.2, 'Gestures +\nPositions', fontsize=7, style='italic', ha='center')
ax.text(5.5, 3.3, 'Associated\nResults', fontsize=8, style='italic', ha='center')

# Add legend for gesture types
legend_y = 0.3
ax.text(1, legend_y, 'Recognized Gestures:', fontsize=9, weight='bold')
ax.text(1.2, legend_y-0.4, 'Peace', fontsize=8)
ax.text(2.5, legend_y-0.4, 'Thumbs Up', fontsize=8)
ax.text(4, legend_y-0.4, 'Stop', fontsize=8)

# Add performance info
ax.text(8.5, legend_y, 'Performance:', fontsize=9, weight='bold')
ax.text(8.5, legend_y-0.4, '15-30 FPS', fontsize=8)

# Title
ax.text(5, 11.7, 'Combined Face and Gesture Recognition System Architecture', 
        ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("System architecture diagram saved as 'system_architecture.png'")
plt.close()

# Also create a simplified flow diagram
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Simplified flow
y_pos = 9
create_box(ax2, 5, y_pos, 2, 0.6, 'Video Frame', color_input, fontsize=11)

y_pos -= 1.2
create_box(ax2, 2.5, y_pos, 2, 0.6, 'Face Detection\n& Recognition', color_face, fontsize=10)
create_box(ax2, 7.5, y_pos, 2, 0.6, 'Hand Detection\n& Classification', color_gesture, fontsize=10)

y_pos -= 1.2
create_box(ax2, 2.5, y_pos, 2, 0.6, 'Face IDs\n& Positions', color_face, fontsize=10)
create_box(ax2, 7.5, y_pos, 2, 0.6, 'Gestures\n& Positions', color_gesture, fontsize=10)

y_pos -= 1.5
create_box(ax2, 5, y_pos, 3, 0.7, 'Spatial Association\n(Distance < 200px)', 
          color_association, fontsize=10)

y_pos -= 1.2
create_box(ax2, 5, y_pos, 2.5, 0.6, 'Person + Gesture\nPairs', color_output, fontsize=10)

y_pos -= 1.2
create_box(ax2, 5, y_pos, 2.5, 0.6, 'Visual + Console\nOutput', color_output, fontsize=10)

# Arrows for simplified flow
create_arrow(ax2, 5, 8.7, 3.5, 8.1)
create_arrow(ax2, 5, 8.7, 6.5, 8.1)
create_arrow(ax2, 2.5, 7.5, 2.5, 6.9)
create_arrow(ax2, 7.5, 7.5, 7.5, 6.9)
create_arrow(ax2, 3.3, 6.3, 4.2, 5.5)
create_arrow(ax2, 6.7, 6.3, 5.8, 5.5)
create_arrow(ax2, 5, 4.65, 5, 4.1)
create_arrow(ax2, 5, 3.5, 5, 2.9)

ax2.text(5, 9.7, 'System Processing Flow', ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('system_flow.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("System flow diagram saved as 'system_flow.png'")
plt.close()

print("\nDiagrams generated successfully!")
print("- system_architecture.png: Detailed architecture with modules")
print("- system_flow.png: Simplified processing flow")
