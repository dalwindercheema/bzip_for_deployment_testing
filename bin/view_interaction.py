import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

def plot_cls_attention_heptad(prot_a,
                              prot_b,
                              attention,
                              prot_names,
                              cmap="vlag"
                              ):
    heptad_start = 15
    prot_a = list(prot_a)
    prot_b = list(prot_b)
    attention = numpy.asarray(attention).flatten()
    residue_attention = attention[1:]

    len_a = len(prot_a)
    len_b = len(prot_b)

    nlen = max(len_a, len_b)

    if len(residue_attention) < nlen:
        raise ValueError(
            f"Attention has {len(residue_attention)} residues, "
            f"but longest sequence has {nlen} residues."
        )

    residue_attention = residue_attention[:nlen]

    valid_attention = residue_attention[
        numpy.isfinite(residue_attention)
    ]

    vmax = valid_attention.max() + 1e-8

    norm = Normalize(
        vmin=0,
        vmax=vmax,
        clip=True
    )

    attention_cmap = plt.get_cmap(cmap)

    heptad = numpy.array(
        list("abcdefg")
    )

    heptad_positions = numpy.array([
        heptad[(i) % 7]
        for i in range(nlen)
    ])

    fig = plt.figure(
        figsize=(len(prot_a)/5, 3),
        constrained_layout=True
    )

    master_gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.25, 1.25], hspace=0.01, figure=fig)
    
    ax_heptad = fig.add_subplot(master_gs[0])
    ax_a = fig.add_subplot(master_gs[1])
    ax_b = fig.add_subplot(master_gs[2], sharex=ax_heptad)

    # ---------------------------------------------------------
    # Heptad annotation
    # ---------------------------------------------------------

    heptad_colors = plt.get_cmap("YlGnBu", 7)

    for c, i in enumerate(range(15, nlen)):
        # Colored heptad block
        ax_heptad.add_patch(
            plt.Rectangle(
                        (i, 0),
                        1,
                        0.35,
                        facecolor=heptad_colors( (c%7)),
                        edgecolor="white",
                        linewidth=0.5
                    )
                )

        # Heptad letter
        ax_heptad.text(
                        i + 0.5,
                        0.45,
                        heptad_positions[c],
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="medium"
                    )

    ax_heptad.set_xlim(heptad_start, nlen)
    ax_heptad.set_ylim(0, 0.5)

    ax_heptad.set_yticks([])
    tick_positions = numpy.arange(heptad_start, nlen, 7)

    ax_heptad.set_xticks(tick_positions + 0.5)

    ax_heptad.set_xticklabels(tick_positions + 1, fontsize=9)

    for spine in ax_heptad.spines.values():
        spine.set_visible(False)
    
    def draw_sequence(ax, sequence):

        seq_len = len(sequence)
        seq_idx = [f"{num}" if num % 10 == 0 else "" for num in range(seq_len)]
        
        for i in range(seq_len):

            att = residue_attention[i]

            # Attention background
            ax.add_patch(
                plt.Rectangle(
                    (i, 0),
                    1,
                    0.4,
                    facecolor=attention_cmap(norm(att)),
                    edgecolor="white",
                    linewidth=0.4
                )
            )

            # Amino acid
            ax.text(
                    i + 0.5,
                    0.5,
                    sequence[i],
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="medium",
                    color="black"
                )
            
            ax.text(
                    i + 0.5,
                    -0.1,
                    seq_idx[i],
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="medium",
                    color="black"
                )

        for i in range(seq_len, nlen):
            ax.add_patch(
                plt.Rectangle((i, 0),
                              1,
                              1,
                              facecolor="white",
                              edgecolor="white",
                              linewidth=0.4
                              )
                        )

        ax.set_xlim(0, nlen)
        ax.set_ylim(0, 0.5)

        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


    draw_sequence(ax_a, prot_a)

    draw_sequence(ax_b, prot_b)
    
    sm = ScalarMappable(norm=norm, cmap=attention_cmap)

    sm.set_array([])
    
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    
    cbar_bottom = pos_b.y0 - 0.1                     # Bottom of the lower axis
    cbar_top = pos_a.y1 - 0.1                       # Top of the upper axis
    cbar_height = cbar_top - cbar_bottom 
    
    cbar_width = 0.01                          # Width of the colorbar
    cbar_left = pos_a.x1 + 0.12 
    

    tick_locations = numpy.linspace(0, vmax, 4)
    
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cax, ticks=tick_locations, format='%.2f')

    cbar.set_label("CLS attention", fontsize=14)
    
    fig.text(
            0.17,
            0.85,
            'Heptad',
            ha="right",
            va="center",
            fontsize=14,
            fontweight="medium",
            color="black"
        )
    
    fig.text(
            0.5,
            -0.05,
            'Residue position',
            ha="center",
            va="center",
            fontsize=14,
            fontweight="medium",
            color="black"
        )
    
    fig.text(
            -0.005,
            0.55,
            prot_names[0],
            ha="right",
            va="center",
            fontsize=14,
            fontweight="medium",
            color="black"
        )
    fig.text(
            -0.005,
            0.18,
            prot_names[1],
            ha="right",
            va="center",
            fontsize=14,
            fontweight="medium",
            color="black"
        )

