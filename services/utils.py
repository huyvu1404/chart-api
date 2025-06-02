def get_table_position(fig, ax, table):
    renderer = fig.canvas.get_renderer()
    bbox = table.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)
    top = bbox_data[:, 1].max()
    left = bbox_data[:, 0].min()
    right = bbox_data[:, 0].max()
    bottom = bbox_data[:, 1].min()
    return top, left, right, bottom

def get_right_edge_x(fig, ax, text_obj):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    bbox_axes = inv.transform((bbox.x1, bbox.y1))  
    right_edge_x = bbox_axes[0]
    return right_edge_x

def get_max_text_widths(rows_data, column_labels=None, font_size=10):

    num_cols = len(column_labels) if column_labels else len(rows_data[0])
    col_max_widths = {i: 0 for i in range(num_cols)}

    full_data = [column_labels] + rows_data if column_labels else rows_data
    for row in full_data:
        for col, val in enumerate(row):
            text = str(val)
            width = len(text) + font_size  
            if width > col_max_widths[col]:
                col_max_widths[col] = width

    total_width = sum(col_max_widths.values())
    for col in col_max_widths:
        col_max_widths[col] = col_max_widths[col] / total_width

    return col_max_widths