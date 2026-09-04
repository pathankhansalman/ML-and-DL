import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_circuit_dag():
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    c_input = "#EBF3FA"
    c_input_border = "#2B5C8F"
    c_probe = "#EDE7F6"
    c_probe_border = "#5E35B1"
    c_attn = "#E8F5E9"
    c_attn_border = "#2E7D32"
    c_mlp = "#FBE9E7"
    c_mlp_border = "#D84315"
    c_steer = "#FFF8E1"
    c_steer_border = "#F57F17"
    c_output = "#E0F2F1"
    c_output_border = "#00695C"

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, badge=None):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=2.0,
            zorder=2
        )
        ax.add_patch(box)
        
        # Text
        if badge:
            ax.text(x + 2, y + h - 3, badge, fontsize=8, fontweight='bold', color=border_color, zorder=3)
            title_y = y + h - 8
        else:
            title_y = y + h - 6
            
        ax.text(x + w/2, title_y, title, fontsize=10.5, fontweight='bold', ha='center', va='center', color="#212121", zorder=3)
        ax.text(x + w/2, y + (h/2) - 3, subtitle, fontsize=8.5, ha='center', va='center', color="#424242", linespacing=1.3, zorder=3)

    # Helper for drawing arrows
    def draw_arrow(start, end, text="", text_offset=(0, 0), style='->', color="#455A64", lw=2.0, ls='-'):
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle=style,
                color=color,
                lw=lw,
                ls=ls,
                shrinkA=5,
                shrinkB=5,
                mutation_scale=15
            ),
            zorder=1
        )
        if text:
            mid_x = (start[0] + end[0]) / 2 + text_offset[0]
            mid_y = (start[1] + end[1]) / 2 + text_offset[1]
            ax.text(mid_x, mid_y, text, fontsize=8, fontweight='bold', color=color,
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85), zorder=4)

    # Title
    ax.text(50, 96, "Causal Circuit Mapping & Steering Architecture (Qwen2.5-0.5B)", 
            fontsize=15, fontweight='bold', ha='center', va='center', color="#1A237E")
    ax.text(50, 92, "Information Flow: From Intent Emergence → Attention Routing → MLP Output Writing & Intervention", 
            fontsize=9.5, ha='center', va='center', color="#5C6BC0", style='italic')

    # --- 1. Inputs (Left Column) ---
    draw_box(4, 62, 24, 22, "Prompt Tokens", 
             "• Order & ADV limits\n• Strategy Verb ('comply' vs 'bypass')\n• Suffix Bridge ('...target venue is')", 
             c_input, c_input_border, badge="STAGE 1: INPUT")

    # --- 2. Intent Formation (Middle Layers) ---
    draw_box(36, 62, 26, 22, "Intent Emergence", 
             "• Layers 6–15\n• Linear Probe Accuracy: 100%\n• Concept cleanly separates\nin residual stream at Verb token", 
             c_probe, c_probe_border, badge="STAGE 2: PROBING")

    # --- 3. Attention Routing (Mid-to-Late Layers) ---
    draw_box(68, 62, 28, 22, "Attention Routing", 
             "• Layers 12–18\n• DLA Contribution: < 4% (Routers)\n• Attention Heads route intent from\nEarly Verb → Final Token Position", 
             c_attn, c_attn_border, badge="STAGE 3: PATCHING")

    # --- 4. Late MLP Output Writers (Bottom Right) ---
    draw_box(68, 18, 28, 26, "MLP Output Writers", 
             "• Layers 20–23 at Final Token\n• DLA Contribution: > 83% (+2.85 logits)\n• Rescue Patching: 100% Causal Effect\n• Directly projects into Unembedding", 
             c_mlp, c_mlp_border, badge="STAGE 4: DLA")

    # --- 5. Final Output (Bottom Center-Left) ---
    draw_box(36, 18, 26, 26, "Vocabulary Output", 
             "• Unembedding Layer (W_U)\n• Clean: P(' exchange') > P(' pools')\n• Evasive: P(' pools') > P(' exchange')", 
             c_output, c_output_border, badge="DECISION")

    # --- 6. Activation Steering Guardrail (Top Right Intervention) ---
    draw_box(4, 18, 24, 26, "Activation Steering", 
             "• PCA Extraction (4.58x vs Mean)\n• Layer 12: Robust (α = 4.0)\n• Layer 21: Sensitive (α = 1.0)\n• Overrides evasive trajectory", 
             c_steer, c_steer_border, badge="GUARDRAIL")

    # --- Connective Causal Arrows ---
    # 1. Inputs -> Probing
    draw_arrow((28, 73), (36, 73), text="Intent Encoded", text_offset=(0, 2))
    
    # 2. Probing -> Attention Routing
    draw_arrow((62, 73), (68, 73), text="Routed Forward", text_offset=(0, 2))

    # 3. Attention Routing -> Late MLPs
    draw_arrow((82, 62), (82, 44), text="Causal Bottleneck (L18-23)", text_offset=(8, 0))

    # 4. Late MLPs -> Vocabulary Output
    draw_arrow((68, 31), (62, 31), text="Writes Output (>83%)", text_offset=(0, 2))

    # 5. Steering Intervention -> Late MLPs / Layer 12
    draw_arrow((28, 31), (36, 31), style='<-', text="Steered (+0.68)", text_offset=(0, 2), color="#E65100", ls="--")
    draw_arrow((16, 44), (72, 44), text="Injected α · v_steer into MLP Inputs", text_offset=(0, 3), color="#E65100", ls="--")

    output_path = "mats_probing/causal_circuit_dag.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Circuit DAG saved successfully: {output_path}")

if __name__ == "__main__":
    draw_circuit_dag()
