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
            marker=dict(size=4, color="blue"),
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
            marker=dict(size=4, color="red"),
        )
    )

    # Set plot title and labels
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    return fig
