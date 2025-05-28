def get_table_position(fig, ax, table):
    renderer = fig.canvas.get_renderer()
    bbox = table.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)
    top = bbox_data[:, 1].max()
    return top

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