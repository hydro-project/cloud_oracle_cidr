## Thank you ChatGPT for helping write this code!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from adjustText import adjust_text

plt.rcParams['text.usetex'] = True

#%matplotlib widget

# Generate 3D space
xmin = 20
ymin = 20
X, Y = np.meshgrid(np.linspace(0, xmin, 30), np.linspace(0, ymin, 30))
xlen = len(X)
ylen = len(Y)

# Example workloads
w0=np.array([3,8])
w1=np.array([6,2])
w2=np.array([9,10])
W=[w0,w1,w2]

# Example planes (z = ax + by + c)
a1, b1, c1 = 1, 1, 1
# a2, b2, c2 = -1, 5, 10
a2, b2, c2 = -5, -5, 100
a3, b3, c3 = 5, -1, 0
Z1 = a1 * X + b1 * Y + c1  # Example plane 1
Z2 = a2 * X + b2 * Y + c2  # Example plane 2
Z3 = a3 * X + b3 * Y + c3  # Example plane 3
# New planes of simple example
#p0 = [15, 80]
#p1 = [80, 15]
#p2 = [95, 95]
#p0 = [1, 15]
#p1 = [15, 1]
#p2 = [16, 16]
p0 = [3, 1]
p1 = [1, 5]
p2 = [p0[0] + p1[0], p0[1] + p1[1]]
P=[p0, p1, p2]
Z1 = p0[0] * X + p0[1] * Y
Z2 = p1[0] * X + p1[1] * Y
Z3 = p2[0] * X + p2[1] * Y
#Z1 = 10 * X + 20 * Y
#Z2 = 20 * X + 10 * Y
#Z3 = 30 * X + 30 * Y

Z_lower = np.minimum(Z1, Z2)
Z_lower = np.minimum(Z_lower, Z3)  # Compute the lower of the two planes

wlColors = ['purple', 'black', 'c']
#z_limit = 2500
#z_limit = 500
z_limit = 120
alpha = 1
alpha_irrelevant = 0.5
tickpadding = -5
labelpadding = -5
titleSpacing = dict(pad=-14, y=0.92)

def gen_fig1(ax, plot_expensive = False, plot_lowest_workload_markers = True, plot_workload_line = True, title = None):
    facecolors = np.empty((xlen, ylen), dtype=object)

    alpha_bottom = 1
    # Plot the z=0 plane with colors corresponding to the lowest plan
    for y in range(ylen):
        for x in range(xlen):
            if Z_lower[x, y] == Z1[x, y]:
                facecolors[x, y] = colors.to_rgba('blue', alpha_bottom)
            elif Z_lower[x, y] == Z2[x, y]:
                facecolors[x, y] = colors.to_rgba('orange', alpha_bottom)
            elif Z_lower[x, y] == Z3[x, y]:
                facecolors[x, y] = colors.to_rgba('green', alpha_bottom)
    ax.plot_surface(X, Y, np.full_like(X,0), facecolors=facecolors, edgecolor=None, shade=False, alpha=0.25, zorder=2)

    # Plot the 3D planes
    def clip_color(*, xlen, ylen, Z, limit, color, lowest, alpha_bottom=1, alpha_top=1):
        return np.array([[colors.to_rgba(color, alpha_bottom if lowest[x,y] == Z[x,y] else alpha_top if Z[x,y] < limit else 0) for y in range(ylen)] for x in range(xlen)], dtype=object)

    # Make Z1 invisible where greater than limit
    facecolorsZ1bottom = clip_color(xlen=xlen, ylen=ylen, Z=Z1, limit=z_limit, alpha_bottom=alpha, alpha_top=0, color="blue", lowest=Z_lower)
    facecolorsZ1top = clip_color(xlen=xlen, ylen=ylen, Z=Z1, limit=z_limit, alpha_bottom=0, alpha_top=alpha, color="blue", lowest=Z_lower)
    facecolorsZ2bottom = clip_color(xlen=xlen, ylen=ylen, Z=Z2, limit=z_limit, alpha_bottom=alpha, alpha_top=0, color="orange", lowest=Z_lower)
    facecolorsZ2top = clip_color(xlen=xlen, ylen=ylen, Z=Z2, limit=z_limit, alpha_bottom=0, alpha_top=alpha, color="orange", lowest=Z_lower)
    facecolorsZ3 = clip_color(xlen=xlen, ylen=ylen, Z=Z3, limit=z_limit, alpha_bottom=alpha_irrelevant, alpha_top=alpha_irrelevant, color="green", lowest=Z_lower)

    lw=0.2
    edgecolor='gray'
    edgecolor=None
    if plot_expensive:
        ax.plot_surface(X, Y, Z3, facecolors=facecolorsZ3, edgecolor=edgecolor, lw=lw, shade=False, zorder=2, label="$p''=\{o,o'\}$")
    ax.plot_surface(X, Y, Z1, facecolors=facecolorsZ1top, edgecolor=edgecolor, lw=lw, shade=False, zorder=4)
    ax.plot_surface(X, Y, Z2, facecolors=facecolorsZ2top, edgecolor=edgecolor, lw=lw, shade=False, zorder=4)
    ax.plot_surface(X, Y, Z1, facecolors=facecolorsZ1bottom, edgecolor=edgecolor, lw=lw, shade=False, zorder=7, label="$p=\{o\}$")
    ax.plot_surface(X, Y, Z2, facecolors=facecolorsZ2bottom, edgecolor=edgecolor, lw=lw, shade=False, zorder=6, label="$p'=\{o'\}$")

    lw=1.5
    # Plot points of workload w0
    for i, w in enumerate(W):
        a = [ (w[0], w[1], w@p) for p in [p0,p1,p2]]
        a = [(x, y, min(z for a, b, z in a if a == x and b == y)) for x, y, z in a]
        
        if plot_lowest_workload_markers:
            # Draw markers at costs of workloads
            ax.scatter(*zip(*a), marker='x', color='r', lw=1.5, zorder=7)
            # Annotate with cost of workload
            x,y,z = a[0]
            x_offset= 0.0
            y_offset= 0.5
            textArgs = dict( label=r'\textbf{[' + f"{x}" + r'\ GB,' + f"{y}" + r'\ gets,\textcent' + f"{z}" + r']}', ha="left", color=wlColors[i], zorder=9, usetex=True, bbox=dict(boxstyle='square,pad=0.01',facecolor='white', edgecolor='white'))
            if i == 0:
                textArgs["ha"] = "right"
                ax.text(x+x_offset,y-y_offset,z+3,s=textArgs.pop("label"), va="bottom", **textArgs)
            elif i == 1:
                ax.text(x+x_offset,y+y_offset,z-3,s=textArgs.pop("label"), va="top", **textArgs)
            elif i == 2:
                ax.text(x+x_offset,y+y_offset,z+3,s=textArgs.pop("label"), va="bottom", **textArgs)

        # Draw vertical lines at coordinates of workloads
        if plot_workload_line:
            x,y,z = a[0]
            ax.plot([x, x], [y, y], [-2, z], marker='', color=wlColors[i], ls='--', lw=1.5, zorder=8)
            ax.plot([x, x], [y, y], [z, z_limit], marker='', color=wlColors[i], ls='--', lw=1.5, zorder=2)
            #ax.plot([w[0], w[0]], [w[1], w[1]], [-2, z_limit], marker='', color=wlColors[i], ls='--', lw=1.5, zorder=6)

    if title is not None:
        ax.set_title(title, **titleSpacing)

    ax.set_xticks(range(0, xmin+1, 5))
    ax.set_yticks(range(0, ymin+1, 5))

    # Set tick padding but keep labels as they are
    ax.tick_params(axis='x', which='major', pad=tickpadding)
    ax.tick_params(axis='y', which='major', pad=tickpadding)
    ax.tick_params(axis='z', which='major', pad=tickpadding+3)
    #ax.grid()

    # Set axis labels
    ax.set_xlabel('Object Size (GB)', labelpad=labelpadding)
    ax.set_ylabel('Get Accesses (req)', labelpad=labelpadding)
    ax.set_zlabel(r'Cost (\textcent)', labelpad=labelpadding+3)

    # Set z-axis limits
    ax.set_zlim(bottom=0, top=z_limit)
    ax.set_ylim(0)
    ax.set_xlim(0)

    return ax



def gen_fig2(ax2, workloads=[0], plot_policy_lines=True, workloads_with_labels=[]):
    if isinstance(workloads_with_labels, bool):
        if workloads_with_labels:
            workloads_with_labels = workloads
        else:
            workloads_with_labels = []
    
    policyColors = ["blue", "orange", "green"]
    
    # Create empty plot with blank marker containing the extra label
    #ax2.plot([], [], ' ', label="Cost of workloads:")
    y_shift = 0.05
    z_shift = 1

    # Plot points of workloads in decision space of current optimizers
    for i in workloads:
        w = W[i]
        points = [
            [1, 0, 1],
            [0, 1, 1],
            [w@p0, w@p1, w@p2]
        ]
        #label=f"$w_{i}$"
        label=r"$\mathbf{"+f"w_{i}:[{w[0]}\,GB,{w[1]}\,gets]"+"}$"
        ax2.scatter(*points, marker='o', color=wlColors[i], zorder=7, label=label)

        for j, (x,y,z) in enumerate(zip(*points)):
            #label = r'$\mathbf{\$_{w_'+f"{i+1},p_{j+1}"+r'}}$\textbf{' + f":[{x},{y},"+r'\textcent'+f"{z}]" + r'}'
            label = r'$\mathbf{\$_{w_'+f"{i+1},p_{j+1}"+r'}}$:\textbf{' +r'\textcent'+f"{z}" + r'}'
            textArgs = dict(ha="right", va="center", usetex=True, zorder=8, bbox=dict(boxstyle='square,pad=0.01',facecolor='white', edgecolor='white'))

            if len(workloads_with_labels) > 1:
                
                textArgs["ha"] = "left"
                textArgs["va"] = "center"

                if j == 0:
                    textArgs["ha"] = "right"
                elif j == 1:
                    if i == 0:
                        textArgs["va"] = "bottom"
                        textArgs["ha"] = "right"
                    elif i == 1:
                        textArgs["va"] = "top"
                    elif i == 2:
                        textArgs["va"] = "bottom"
                elif j == 2:
                    if i == 1:
                        textArgs["va"] = "bottom"
                    elif i == 2:
                        textArgs["va"] = "top"
            
            if textArgs["ha"] == "right":
                y -= y_shift
            elif textArgs["ha"] == "left":
                y += y_shift

            if textArgs["va"] == "bottom":
                z -= z_shift
            elif textArgs["va"] == "top":
                z += z_shift

            if i in workloads_with_labels:
                t = ax2.text(x, y, z, s=label, color=wlColors[i], **textArgs)
                

    if plot_policy_lines:
        #ax2.plot([], [], ' ', label="Cost of policies:")
        ax2.plot([1, 1], [0, 0], [-2, z_limit], marker='', color='blue', ls='--', lw=1, zorder=4)
        ax2.plot([0, 0], [1, 1], [-2, z_limit], marker='', color='orange', ls='--', lw=1, zorder=4)
        ax2.plot([1, 1], [1, 1], [-2, z_limit], marker='', color='green', ls='--', lw=1, zorder=4)

        textArgs = dict(ha="left", va="center", usetex=True, zorder=9, bbox=dict(boxstyle='square,pad=0.01',facecolor='white', edgecolor='white'))
        ax2.text(1, 0+y_shift, 4, color='blue', s="$\mathbf{p_1}$:$\mathbf{\{o_1\}}$", **textArgs)
        ax2.text(0, 1+0.07, 4, color='orange', s="$\mathbf{p_2}$:$\mathbf{\{o_2\}}$", **textArgs)
        ax2.text(1, 1+y_shift, 4, color='green', s="$\mathbf{p_3}$:$\mathbf{\{o_1,o_2\}}$", **textArgs)
    # Move legend to the right
    ax2.legend(loc='center left', bbox_to_anchor=(0.7, 0.6))

    workloadString = ",".join([f"w_{i+1}" for i in workloads])
    ax2.set_title("Concrete costs of workloads", **titleSpacing)

    # Set axis labels and ticks of ax2
    ax2.tick_params(axis='x', which='major', pad=tickpadding)
    ax2.tick_params(axis='y', which='major', pad=tickpadding)
    ax2.tick_params(axis='z', which='major', pad=tickpadding+3)
    ax2.set_xlabel(r"$\mathbf{b_1}$: binary choice for object store $\mathbf{o_1}$", labelpad=labelpadding)
    ax2.set_ylabel(r"$\mathbf{b_2}$: binary choice for object store $\mathbf{o_2}$", labelpad=labelpadding)
    ax2.set_zlabel(r"Cost (\textcent)", labelpad=labelpadding+3)
    ax2.set_xticks([0,1])
    ax2.set_yticks([0,1])
    #ax2.set_xticklabels(["No", "Yes"])
    #ax2.set_yticklabels(["No", "Yes"])
    ax2.set_zlim([0, z_limit])
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_xlim(-0.6, 1.6)
    #ax2.set_title("Cost-Placement Space")
    return ax2

figsize=(5.5, 5)
view = dict(elev=19, azim=19, roll=0)
view = dict(elev=7, azim=27, roll=0)
sav_pad = 0.4

fig5 = plt.figure(figsize=figsize)
ax5 = fig5.add_subplot(111, projection='3d')
ax5.computed_zorder=False
ax5 = gen_fig2(ax5, workloads=[0,1,2], workloads_with_labels=[0])
ax5.view_init(**view)
fig5.savefig(f'intuition-current-approach.pdf')

# Create 3D plot
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder=False
ax  = gen_fig1(ax, title=r"Costs of placement policies $\mathbf{p_1,p_2,p_3}$", plot_expensive=False, plot_workload_line=True)
ax.view_init(**view)
fig.savefig('intuition-min.pdf',bbox_inches='tight', pad_inches=sav_pad)

fig4 = plt.figure(figsize=figsize)
ax4 = fig4.add_subplot(111, projection='3d')
ax4.computed_zorder=False
ax4 = gen_fig1(ax4, plot_expensive=True, plot_workload_line=False, plot_lowest_workload_markers=False)
ax4.view_init(**view)
fig4.savefig('intuition-enumeration.pdf',bbox_inches='tight', pad_inches=sav_pad)

fig2 = plt.figure(figsize=figsize)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.computed_zorder=False
ax2 = gen_fig1(ax2, title=r"Costs of placement policies $\mathbf{p_1,p_2,p_3}$", plot_expensive=True, plot_workload_line=True, plot_lowest_workload_markers=True)
ax2.view_init(**view)
fig2.savefig('intuition-all.pdf',bbox_inches='tight', pad_inches=sav_pad)

for w in [[0],[1],[2]]:
    fig3 = plt.figure(figsize=figsize)
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.computed_zorder=False
    ax3 = gen_fig2(ax3, workloads=w)
    ax3.view_init(**view)
    fig3.savefig(f'intuition-current-approach_{"-".join([f"w_{i}" for i in w])}.pdf')

# Show the plot
plt.show()
