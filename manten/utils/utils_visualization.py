def visualize_2_pos_traj(pred_pos, gt_pos):
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add predicted positions
    fig.add_trace(
        go.Scatter3d(
            x=pred_pos[:, 0],
            y=pred_pos[:, 1],
            z=pred_pos[:, 2],
            mode="markers+lines",
            name="Predicted",
            marker={"size": 4, "color": "blue"},
        )
    )

    # Add ground truth positions
    fig.add_trace(
        go.Scatter3d(
            x=gt_pos[:, 0],
            y=gt_pos[:, 1],
            z=gt_pos[:, 2],
            mode="markers+lines",
            name="Ground Truth",
            marker={"size": 4, "color": "red"},
        )
    )

    # Set plot title and labels
    fig.update_layout(scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"})

    return fig


def visualize_image(images, labels=None, grid_edge=5):
    import einops
    import torch
    import wandb

    images = images[:grid_edge * grid_edge]
    # images: (B, C, H, W) to (B, C, H, W) with B = grid_edge * grid_edge, must pad accordingly
    if images.shape[0] < grid_edge * grid_edge:
        pad = torch.zeros(grid_edge * grid_edge - images.shape[0], *images.shape[1:], dtype=images.dtype)
        images = torch.cat([images, pad], dim=0)

    image = einops.rearrange(images, "(nh nw) c h w -> c (nh h) (nw w)", nh=grid_edge, nw=grid_edge)

    if labels is not None:
        caption = f"Labels: {labels[:grid_edge * grid_edge]}"
    else:
        caption = None

    return wandb.Image(image, caption=caption)
