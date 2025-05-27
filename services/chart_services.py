import random
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import plotly.graph_objects as go
from io import BytesIO
from wordcloud import WordCloud
from fastapi.responses import StreamingResponse
from models import TableRequest, SanKeyChartRequest, PieChartRequest, BarChartRequest, LineChartRequest, WordCloudRequest
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
            width = len(text) + 10  
            if width > col_max_widths[col]:
                col_max_widths[col] = width

    total_width = sum(col_max_widths.values())
    for col in col_max_widths:
        col_max_widths[col] = col_max_widths[col] / total_width

    return col_max_widths

async def generate_bar_chart(request: BarChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")

    x = request.x
    y = request.y
    len_x = len(x)

    # Nếu chỉ có 1 giá trị x, thêm đệm trái và phải
    if len_x == 1:
        x = [""] + x + [""]
        for i in range(len(y)):
            if isinstance(y[i], list):
                y[i] = [0] + y[i] + [0]
        len_x = len(x)  
  
    x_pos = np.arange(len_x)
    bottom = np.zeros(len_x)

    fig, ax = plt.subplots(figsize=(len_x + 3, 5))
    bar_width = 0.2
    max_value = 0

    if isinstance(y[0], list):
        for i, y_i in enumerate(y):
            if isinstance(y_i, list):
                max_y = max(y_i)
                max_value = max(max_value, max_y)
                ax.bar(x_pos, y_i, width=bar_width, bottom=bottom, color=request.colors[i], label=request.labels[i])
                bottom += y_i
    else:
        max_value = max(y)
        ax.bar(x_pos, y, color=request.colors)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)

    ylim = int(np.ceil(max_value / 10) * 10)
    step = int(np.ceil(ylim / 50) * 10)
    if step == 0 or step == 1:
        step = 1
    ax.set_ylim(-step, ylim)

    grid_yticks = np.arange(-step, ylim + step, step)
    ax.set_yticks(grid_yticks)
    ax.set_yticklabels([str(tick) if tick >= 0 else None for tick in grid_yticks])

    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    fig.suptitle(request.title, x=0.01, y=0.98, ha='left', fontsize=16, weight='bold')
    fig.subplots_adjust(top=0.88)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='lower center', bbox_to_anchor=(1, 1), frameon=False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

async def generate_top_social_posts(request: BarChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    
    categories = request.x
    total_engagements, sentiment_score = request.y
    colors = request.colors

    fig, ax = plt.subplots(figsize=(14, 8)) 
    x = np.arange(len(categories))
    bar_width = 0.25  
    ylim = int(np.ceil(max(total_engagements) / 10) * 10)
    step = int(ylim / 2)
    if step == 0 or step == 1:
        step = 1
    ax.bar(x, total_engagements, width=bar_width, color=colors[0], label='Total Engagements')
    scale = 100 / step
    scaled_sentiment_score = [-s / scale for s in sentiment_score]
    bars2 = ax.bar(x, scaled_sentiment_score, width=bar_width, color=colors[1], label='Sentiment Score')

    for i, bar in enumerate(bars2):
        height = -sentiment_score[i] / scale
        if height < 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height - 1, f'{int(abs(height * scale))}%', ha='center', va='top', color='black')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height - 1, '0%', ha='center', va='top', color='red')

    ax.set_ylim(-ylim, ylim)
    grid_yticks = np.arange(-step, ylim + step, step)
    ax.set_yticks(grid_yticks)
    ax.set_yticklabels([str(tick) if tick >= 0 else None for tick in grid_yticks])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    ax.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

async def generate_pie_chart(request: PieChartRequest):
    if not request.values:
        return StreamingResponse(BytesIO(), media_type="image/png")
    total = sum(request.values)
    sizes = request.values
    labels = []
    percentages = []
    for label, value in zip(request.labels, request.values):
        if value > 0:
            percentage = (value / total) * 100
        else:
            percentage = 0
        percentages.append(percentage)
        labels.append(f"{label} ({value})\n{percentage:.2f}%")
    
    colors = request.colors

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(sizes, colors=colors, wedgeprops=dict(width=0.3, edgecolor='white'), startangle=90)

    ax.text(0, 0.1, f"{total:,}", fontsize=20, weight="bold", ha='center')
    ax.text(0, -0.1, "Mentions", fontsize=12, color='gray', ha='center')

    legend_labels = [f"{percentage:.2f}%  {label}  ({value})" for percentage, label, value in zip(percentages, request.labels, request.values)]
    legend = ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_color("#555")  
    fig.text(
        0.01, 0.8,  
        request.title,
        ha='left',
        va='top',
        fontsize=14,
        fontweight='bold',
        color='#333'  
    )


    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")

async def generate_wordcloud(request: WordCloudRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")
    word_freq = {item["key"]: item["doc_count"] for item in request.data}
    color = 'Greens_r' if 'Positive' in request.title else 'Reds_r'


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(BASE_DIR, 'fonts', 'DejaVuSans.ttf')
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=color,
        font_path=font_path 
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")

async def generate_line_chart(request: LineChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    fig, ax = plt.subplots()
    if isinstance(request.y[0], list):
        for i, y in enumerate(request.y):
            if isinstance(y, list):
                ax.plot(request.x[:len(y)], y, color=request.colors[i], label=request.labels[i])
    else:
        ax.plot(request.x[:len(y)], request.y, color=request.colors, label=request.labels[0])
    
    fig.suptitle(request.title, x=0.01, ha='left', fontsize=14, weight='bold')
    ax.set_xlabel(request.xlabel)
    ax.set_ylabel(request.ylabel)
    ax.tick_params(axis='y', length=0)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

async def generate_trend_chart(request: LineChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    
    current = request.y[0] if len(request.y) > 0 else []
    previous = request.y[1] if len(request.y) > 1 else []

    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.88)  

    color = ["#61ff00", "#d9f9c5"] if "Positive" in request.title else ["#FF5733", "#f9b6aa"]

    ax.plot(request.x[:len(current)], previous[:len(current)], color=color[1], label=request.labels[0], linestyle='--', marker='o')
    ax.plot(request.x[:len(current)], current, color=color[0], label=request.labels[1], marker='o')

    total_current = sum(current)
    total_previous = sum(previous)
    percentage_change = ((total_current - total_previous) / total_previous) * 100 if total_previous > 0 else 0

    emoji = "😞" if "Negative" in request.title else "😃"
    arrow = "⬆️" if total_current > total_previous else "⬇️"

    fig.suptitle(request.title, x=0.01, y=1.02, ha='left', fontsize=14, weight='bold')

    t1 = fig.text(
        0.1, 0.93,
        f"{emoji} {total_current}",
        ha='left', va='top',
        fontsize=14, fontweight='bold',
        color=color[0]
    )

    renderer = fig.canvas.get_renderer()
    bbox = t1.get_window_extent(renderer=renderer)
    inv = fig.transFigure.inverted()
    new_pos = inv.transform((bbox.x1, bbox.y1))[0] + 0.01 

    fig.text(
        new_pos, 0.92,
        f"{arrow} {abs(percentage_change):.2f}% (compare to last period)",
        ha='left', va='top',
        fontsize=10, fontweight='bold',
        color=color[0]
    )

    ax.set_xlabel(request.xlabel)
    ax.set_ylabel(request.ylabel)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

async def generate_sankey_chart(request: SanKeyChartRequest):
    sources = request.data["sources"] if request.data.get("sources") else []
    sentiments = request.data["sentiments"] if request.data.get("sentiments") else []
    topics = request.data["topics"] if request.data.get("topics") else []
    
    if not sources and not sentiments and not topics:
        return StreamingResponse(BytesIO(), media_type="image/png")
    labels = sources + sentiments + topics
    node_indices = {label: idx for idx, label in enumerate(labels)}
    
    source_list = []
    target_list = []
    value_list = []
    custom_data = []

    for link in request.data["links"]:
        if 'source' in link and 'sentiment' in link:
            source_name = link['source']
            sentiment_name = link['sentiment']
            value = link['value']
            source_list.append(node_indices[source_name])
            target_list.append(node_indices[sentiment_name])
            value_list.append(value)

    for link in request.data["links"]:
        if 'sentiment' in link and 'topic' in link:
            sentiment_name = link['sentiment']
            topic_name = link['topic']
            value = link['value']
            source_list.append(node_indices[sentiment_name])
            target_list.append(node_indices[topic_name])
            value_list.append(value)

    total_value = sum(value_list)

    node_values = [0] * len(labels)  
    for src, tgt, val in zip(source_list, target_list, value_list):
        node_values[src] += val  
        node_values[tgt] += val  
    
    updated_labels = []
    for i, label in enumerate(labels):
        value = node_values[i]
        percentage = (value / total_value) * 100 if total_value > 0 else 0
        updated_labels.append(f"{label} {percentage:.2f}% ({value})")

    for value in value_list:
        custom_data.append(f"{value} mention")

    sentiment_colors = {
        "Positive": "#99ff33",
        "Neutral": "#c9cac8",
        "Negative": "#f53105"
    }

    random.seed(40)
    def get_random_color():
        return "#%06x" % random.randint(0, 0xFFFFFF)

    node_colors = [sentiment_colors.get(label, get_random_color()) for label in labels]

    def hex_to_rgba(hex_color, alpha):
        rgb = mcolors.to_rgb(hex_color)  
        return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"

    link_colors = [hex_to_rgba(node_colors[src], 0.2) for src in source_list]        
    link_hovercolors = [hex_to_rgba(node_colors[src], 0.6) for src in source_list]  

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=updated_labels,
            color=node_colors,  
        ),
        link=dict(
            source=source_list,
            target=target_list,
            value=value_list,
            color=link_colors,
            hovercolor=link_hovercolors,
            customdata=custom_data,
            hovertemplate="%{customdata}<extra></extra>"
        )
    )])

    fig.update_layout(
        width=1500,    
        height=1000,
        title_text=f"<b>{request.title}</b>",  
        title_x=0.1, 
        font_size=10
    )

    buf = BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

async def generate_top_sources(request: TableRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")
    
    all_data = request.data
    cols = 3
    num_tables = len(all_data)
    rows = math.ceil(num_tables / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    axes_flat = [ax for row in axes for ax in row]

    for j in range(len(all_data), len(axes_flat)):
        axes_flat[j].set_visible(False)

    for i, data in enumerate(all_data):
        ax = axes_flat[i]
        ax.axis("off")

        title = data.title
        rows_data = data.rows
        net_sentiment_score = data.net_sentiment_score if data.net_sentiment_score else 0
        total_mention = data.total if data.total else 0
        total_score = sum(r[1] for r in rows_data) or 1

        table_data = [
            [r[0][:20] + '...' if len(r[0]) > 20 else r[0], f"{round(r[1] / total_score * 100, 2)}% ({r[1]})"] for r in rows_data
        ]
        column_labels = data.column_labels if data.column_labels else None
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels if column_labels else None,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.set_height(0.1) 
            if column_labels:
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#cdf8f5')
            if col == 0:
                cell.set_text_props(ha='left', weight='bold', color='#666666')
                cell.set_facecolor('#c2f9f5')
                cell.set_width(0.7)
            elif col == 1:
                cell.set_text_props(ha='right', weight='bold', color='#666666')
                cell.set_facecolor('#dffcfa')
                cell.set_width(0.25)

        top = get_table_position(fig, ax, table)
        # color light blue
        net_sentiment_score_color = '#61ff00' if net_sentiment_score >= 0 else '#f53105'
        ax.text(0.5, top + 0.40, title, ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, top + 0.35, "Total Mention", ha='left', fontsize=12, transform=ax.transAxes, fontweight='bold', color='#666666')
        ax.text(0.05, top + 0.25, f"{total_mention}", ha='left', fontsize=12, fontweight='bold', transform=ax.transAxes, color=net_sentiment_score_color)
        ax.text(0.05, top + 0.15, "Net Sentiment Score", ha='left', fontsize=12, transform=ax.transAxes, fontweight='bold', color='#666666')
        ax.text(0.05, top + 0.05, f"{net_sentiment_score}%", ha='left', fontsize=12, fontweight='bold', transform=ax.transAxes, color=net_sentiment_score_color)
        ax.axis("off")
    
    plt.subplots_adjust(hspace=0.8)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")

async def generate_overview(request: TableRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")

    data = request.data
   
    fig, ax = plt.subplots(figsize=(8, 2))

    title = data.title
    rows_data = data.rows

    column_labels = data.column_labels if data.column_labels else None
    table = ax.table(
        cellText=rows_data,
        colLabels=column_labels,
        cellLoc='center',
        loc='center'
    )
    col_widths = get_max_text_widths(rows_data, column_labels, font_size=10)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        cell.set_height(0.2) 
        if column_labels and row == 0:
            cell.set_facecolor('#05c2f5')
            cell.set_text_props(weight='bold', color='#f4f6f7')        
        else:
            cell.set_text_props(weight='bold', color='#666666')
            
    top = get_table_position(fig, ax, table)
    if title:
        ax.text(0.01, top + 0.1, title, ha='left', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis("off")
        
    plt.subplots_adjust(hspace=0.8)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")


async def generate_brand_attribute(request: TableRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")

    data = request.data
    title = data.title
    rows_data = data.rows
    fig, ax = plt.subplots(figsize=(len(data.rows)*0.5 + 2, len(data.rows) * 0.2 + 2))
    column_labels = data.column_labels if data.column_labels else None

    table = ax.table(
        cellText=rows_data,
        colLabels=column_labels,
        cellLoc='center',  
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    col_widths = get_max_text_widths(rows_data, column_labels, font_size=10)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        cell.set_height(0.1) 
        if column_labels and row == 0:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold', color='black')  
        elif col == 0:
            cell.set_text_props(ha='left', weight='bold', color='#666666')           
        elif col == 1:
            cell.set_text_props(ha='right', weight='bold', color='#666666')       
        else:
            cell.set_text_props( color='#666666')
            
    top = get_table_position(fig, ax, table)
    ax.text(0.01, top + 0.1, title, ha='left', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis("off")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")
