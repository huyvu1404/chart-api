import random
import math
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

def get_normalized_col_widths(table):
    font_prop = FontProperties(size=10)
    col_widths = {}
    for (row, col), cell in table.get_celld().items():
        txt = str(cell.get_text().get_text())
        est_width = font_prop.get_size_in_points() * len(txt) * 0.6  
        if col not in col_widths or est_width > col_widths[col]:
            col_widths[col] = est_width

    total_width = sum(col_widths.values())
    for col in col_widths:
        col_widths[col] /= total_width

    return col_widths

async def generate_bar_chart(request: BarChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    
    bottom = np.zeros(len(request.x))
    fig, ax = plt.subplots(figsize=(14, 8)) 
    bar_width = 0.25
    max_value = 0
    if isinstance(request.y[0], list):
        for i, y in enumerate(request.y):
            if isinstance(y, list):
                max_y = max(y)
                max_value = max(max_value, max_y)
                ax.bar(request.x, y, width=bar_width, bottom=bottom, color=request.colors[i], label=request.labels[i])
                bottom += y
    else:
        ax.bar(request.x, request.y, color=request.colors)
    
    ylim = int(np.ceil(max_value / 10) * 10)
    step = int(np.ceil(ylim / 50) * 10) 
    ax.set_ylim(-step, ylim)
    grid_yticks = np.arange(-step, ylim + step, step)
    ax.set_yticks(grid_yticks)
    ax.set_yticklabels([str(tick) if tick >= 0 else None for tick in grid_yticks])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    fig.suptitle(request.title, x=0.01, ha='left', fontsize=14, weight='bold')
    ax.set_xlabel(request.xlabel)
    ax.set_ylabel(request.ylabel)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.legend()  
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
    ax.legend(wedges, legend_labels, title=request.title, loc="center left", bbox_to_anchor=(1, 0.5))

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
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='/fonts/DejaVuSans.ttf' 
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
    ax.plot(request.x[:len(previous)], previous, color=request.colors[0], label=request.labels[0], linestyle='--', marker='o')
    ax.plot(request.x[:len(current)], current, color=request.colors[1], label=request.labels[1], marker='o')

    total_current = sum(current)
    total_previous = sum(previous)
    percentage_change = ((total_current - total_previous) / total_previous) * 100 if total_previous > 0 else 0


    emoji = "😞" if "Negative" in request.title else "😃"
    arrow = "⬆️" if total_current > total_previous else "⬇️"

    fig.suptitle(request.title, x=0.01, ha='left', fontsize=14, weight='bold')
    fig.text(0.1, 0.91, f"{total_current} {emoji} {arrow}{abs(percentage_change):.2f}% (compare to last period)", ha='left', va='top', fontsize=12)
    ax.set_xlabel(request.xlabel)
    ax.set_ylabel(request.ylabel)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
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

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))

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
            if column_labels:
                if row == 0:
                    cell.set_text_props(weight='bold')
            if col == 0:
                cell.set_text_props(ha='left')
                cell.set_width(0.7)
                cell.set_height(0.1)
            elif col == 1:
                cell.set_text_props(ha='right')
                cell.set_width(0.2)
                cell.set_height(0.1)

        top = get_table_position(fig, ax, table)
        ax.text(0.5, top + 0.40, title, ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, top + 0.35, "Total Mention", ha='left', fontsize=12, transform=ax.transAxes)
        ax.text(0.05, top + 0.25, f"{total_mention}", ha='left', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, top + 0.15, "Net Sentiment Score", ha='left', fontsize=12, transform=ax.transAxes)
        ax.text(0.05, top + 0.05, f"{net_sentiment_score}%", ha='left', fontsize=12, fontweight='bold', transform=ax.transAxes)
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
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    col_widths = get_normalized_col_widths(table)
    for (row, col), cell in table.get_celld().items():
        if col in col_widths:
            cell.set_width(col_widths[col])
    for (row, col), cell in table.get_celld().items():
        if column_labels:
            if row == 0: 
                cell.set_text_props(weight='bold')

    renderer = fig.canvas.get_renderer()
    bbox = table.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)
    top_y = bbox_data[:,1].max()
    ax.text(0.07, top_y + 0.1, title, ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
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
    fig, ax = plt.subplots(figsize=(8, 3))

    title = data.title
    rows_data = data.rows
    column_labels = data.column_labels if data.column_labels else None

    table = ax.table(
        cellText=rows_data,
        colLabels=column_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    col_widths = get_normalized_col_widths(table)
    for (row, col), cell in table.get_celld().items():
        if col in col_widths:
            cell.set_width(col_widths[col])

    if column_labels:
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')

    top = get_table_position(fig, ax, table)
    ax.text(0.01, top + 0.1, title, ha='left', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")
