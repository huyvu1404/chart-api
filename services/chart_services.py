import random
import math
import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from io import BytesIO
from wordcloud import WordCloud
from fastapi.responses import StreamingResponse
from models import TableRequest, SanKeyChartRequest, PieChartRequest, BarChartRequest, LineChartRequest, WordCloudRequest
from .utils import get_table_position, get_max_text_widths, get_right_edge_x

async def generate_bar_chart(request: BarChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")

    x = request.x
    y = request.y

    x = [""] + x + [""]
    x_pos = np.arange(len(x))
    bottom = np.zeros(len(x))

    fig, ax = plt.subplots(figsize=(len(x) + 3, 5))
    bar_width = 0.2
    max_value = 0

    if isinstance(y[0], list):
        for i, y in enumerate(y):
            if isinstance(y, list):
                max_y = max(y)
                y = [0] + y + [0]
                max_value = max(max_value, max_y)
                ax.bar(x_pos, y, width=bar_width, bottom=bottom, color=request.colors[i], label=request.labels[i])
                bottom += y
    else:
        max_value = max(y)
        y = [0] + y + [0]
        ax.bar(x_pos, y, width=bar_width, color=request.colors)

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

async def generate_conversation_breakdown(request: BarChartRequest):
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    if not isinstance(request.y[0], list):
        return StreamingResponse(BytesIO(), media_type="image/png")
    colors = request.colors if request.colors else ["#D3D3D3", "#4CAF50", "#F44336"]
    labels = request.labels if request.labels else ["Neutral", "Positive", "Negative"]
    x = [""] + request.x + [""]
    x_pos = np.arange(len(x))
    bottom = np.zeros(len(x))

    fig, ax = plt.subplots(figsize=(len(x) + 3, 5))
    bar_width = 0.2
    max_value = 0

    for i, y in enumerate(request.y):
        if isinstance(y, list):
            max_y = max(y)
            y = [0] + y + [0]
            max_value = max(max_value, max_y)
            ax.bar(x_pos, y, width=bar_width, bottom=bottom, color=colors[i], label=labels[i])
            bottom += y

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
    legend_patches = [
        Patch(color=colors[i], label=labels[i]) for i in range(len(labels))
    ]
    ax.legend(legend_patches, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=len(labels), frameon=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

async def generate_top_social_posts(request: BarChartRequest):
    if not request:
        return StreamingResponse(BytesIO(), media_type="image/png")
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    
    num_bars = len(request.x)
    sites_name = request.x
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=18)) for label in sites_name]
    sites_name = [" "] + wrapped_labels + [" "]
    interactions, sentiment_score = request.y
    interactions = [0] + interactions + [0]
    sentiment_score = [0] + sentiment_score + [0]
    colors = request.colors or ["#FFA500", "#00CED1"]
    labels = request.labels or [
        "Tổng tương tác = Tổng yêu thích + Tổng chia sẻ + Tổng bình luận",
        "Chỉ số cảm xúc = (Tích cực - Tiêu cực) / (Tích cực + Tiêu cực)*100%"
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(sites_name))
    bar_width = 0.25
    ylim = max(10, int(np.ceil(max(interactions) / 10) * 10))
    step = max(1, int(ylim / 2))
    scale_factor = 100 / step
    negative_height = step / 5 if num_bars > 2 else min(step / 3, step / num_bars)

    ax.bar(x, interactions, width=bar_width, color=colors[0], label=labels[0])

    bar_heights = []
    bar_widths = []
    bar_bottoms = []
    sentiment_score_colors = []

    for i, s in enumerate(sentiment_score):
        if i == 0 or i == len(sentiment_score) - 1:
            bar_heights.append(0)
            bar_bottoms.append(0)
            bar_widths.append(0.25)
            sentiment_score_colors.append("white")
        elif s > 0:
            bar_heights.append(-s / scale_factor)
            bar_bottoms.append(0)
            bar_widths.append(0.25)
            sentiment_score_colors.append(colors[1])
        else:
            if s == 0:
                bar_widths.append(0.25)
            else:
                bar_widths.append(0.3)
            bar_heights.append(-negative_height)
            bar_bottoms.append(-negative_height / 4)
            sentiment_score_colors.append("red")
        

    bars2 = ax.bar(
        x, bar_heights, width=bar_widths, bottom=bar_bottoms,
        color=sentiment_score_colors, label=labels[1]
    )

    for i, (bar, score) in enumerate(zip(bars2, sentiment_score)):
        if 0 < i < len(sentiment_score) - 1:
            height = bar_heights[i]
            fontsize = 60 / len(sites_name)
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height - negative_height / 3,
                        f'{score}%', ha='center', va='top', color='black', weight='bold', fontsize=fontsize)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height * 5 / 8,
                        f'{score}%', ha='center', va='top', color='white', weight='bold', fontsize=fontsize)

    ax.set_ylim(-ylim * 0.8, ylim)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(-step, ylim + step, step))
    ax.set_xticklabels(sites_name)
    ax.set_yticklabels([str(tick) if tick >= 0 else None for tick in np.arange(-step, ylim + step, step)]) 

    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    legend_patches = [
        Patch(color=colors[0], label=labels[0]),
        Patch(color=colors[1], label=labels[1])
    ]
    ax.legend(legend_patches, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=len(labels), frameon=False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
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

async def generate_channel_breakdown(request: PieChartRequest):
    return await generate_pie_chart(request)

async def generate_sentiment_breakdown(request: PieChartRequest):
    return await generate_pie_chart(request)

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
    if not request:
        return StreamingResponse(BytesIO(), media_type="image/png")
    if not request.x or not request.y:
        return StreamingResponse(BytesIO(), media_type="image/png")
    
    previous = request.y[0] if len(request.y) > 1 else []
    current = request.y[1] if len(request.y) > 0 else []
 
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
        0.1, 0.94,
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
        new_pos, 0.93,
        f"{arrow} {abs(percentage_change):.2f}% (compare to last period)",
        ha='left', va='top',
        fontsize=10, fontweight='bold',
        color=color[0]
    )
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

async def generate_topic_distribution(request: SanKeyChartRequest):
    sources = request.data.get("sources") if request.data.get("sources") else {}
    sentiments = request.data.get("sentiments")if request.data.get("sentiments") else []
    topics = request.data.get("topics") if request.data.get("topics") else []
    
    if not sources and not sentiments and not topics:
        return StreamingResponse(BytesIO(), media_type="image/png")
    source_labels = [source["source"] for source in sources]
    topic_labels = [topic["topic"] for topic in topics]
    labels = source_labels + sentiments + topic_labels
    node_indices = {label: idx for idx, label in enumerate(labels)}
    
    source_list = []
    total_source = [source["total"] for source in sources]
   
    # create a dict source - total and topic - total named labels 
    
    sources_dict = {source["source"]: source["total"] for source in sources} 
    
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

    total_value = sum(total_source)
    node_values = [0] * len(labels)  
    for src, tgt, val in zip(source_list, target_list, value_list):
        node_values[src] += val  
        node_values[tgt] += val  
    
    updated_labels = []
    for i, label in enumerate(labels):
        value = sources_dict.get(label) #or node_values[i]
        if value:
            percentage = (value / total_value) * 100 if total_value > 0 else 0
            updated_labels.append(f"{label} {percentage:.2f}% ({value})")
        else:
            updated_labels.append(label)

    for value in value_list:
        custom_data.append(f"{value} mention")

    sentiment_score_colors = {
        "Positive": "#99ff33",
        "Neutral": "#c9cac8",
        "Negative": "#f53105"
    }

    random.seed(10)
    def get_random_color():
        return "#%06x" % random.randint(0, 0xFFFFFF)

    node_colors = [sentiment_score_colors.get(label, get_random_color()) for label in labels]

    def hex_to_rgba(hex_color, alpha):
        rgb = mcolors.to_rgb(hex_color)  
        return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"

    link_colors = [hex_to_rgba(node_colors[src], 0.2) for src in source_list]        
    link_hovercolors = [hex_to_rgba(node_colors[src], 0.6) for src in source_list]  

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,
            thickness=15,
            line=dict(color="rgba(0,0,0,0)", width=0),
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for ax in axes_flat[len(all_data):]:
        ax.axis("off")

    for i, data in enumerate(all_data):
        ax = axes_flat[i]
        title = data.title
        rows_data = data.rows
        score = data.score if data.score else 0
        total_mention = data.total if data.total else 0
        percentage_change = data.percentage if data.percentage else 0
        total_score = sum(r[1] for r in rows_data) or 1
        if percentage_change > 0:
            percentage_str = f" ⬆️ {(percentage_change):.2f}%"
            text_color = '#61ff00'
        elif percentage_change < 0:
            percentage_str = f" ⬇️ {abs(percentage_change):.2f}%"
            text_color = '#f53105'
        else:
            percentage_str = ""
            text_color = '#666666'

        table_data = [
            [r[0][:20] + '...' if len(r[0]) > 20 else r[0], f"{round(r[1] / total_score * 100, 2)}% ({r[1]})"] for r in rows_data
        ]
        headers = data.headers if data.headers else None
        table = ax.table(
            cellText=table_data,
            colLabels=headers if headers else None,
            cellLoc='center',
            loc='best'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.set_height(0.1) 
            if headers:
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

        top, left, right, bottom = get_table_position(fig, ax, table)

        ax.text(0.5, top + 0.40, title, ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
        total_mention_text = ax.text(0.05, top + 0.30, "Total Mention", ha='left', fontsize=12,
                   transform=ax.transAxes, fontweight='bold', color='#a7a7a7')
        right_edge_x = get_right_edge_x(fig, ax, total_mention_text)
        ax.text(right_edge_x, top + 0.25, f"{total_mention}", ha='right', fontsize=12,
                fontweight='bold', transform=ax.transAxes, color="#2c2c2c")
        ax.text(right_edge_x, top + 0.20, f"{percentage_str}", ha='right', fontsize=12,
                color=text_color, fontweight='bold', transform=ax.transAxes)
        net_sentiment_text = ax.text(0.05, top + 0.10, "Net Sentiment Score", ha='left', fontsize=12, transform=ax.transAxes, fontweight='bold', color='#2c2c2c')
        right_edge_x = get_right_edge_x(fig, ax, net_sentiment_text)
        ax.text(right, top + 0.05, f"{score}%", ha='right', fontsize=12, fontweight='bold', transform=ax.transAxes, color='#a7a7a7')
        ax.axis("off")
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")

async def generate_overview(request: TableRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")

    data = request.data
   
    fig, ax = plt.subplots(figsize=(8, 2))
    
    title = data.title
    current, last = data.rows
    rows_data = [current]
    change_row = []
    colors = []
    for current_value, last_value in zip(current, last):
        diff = (current_value - last_value)/ last_value * 100 if last_value != 0 else 0
        if diff > 0:
            row_color = '#61ff00'  
            arrow = '⬆️'
        elif diff < 0:
            row_color = '#f53105'
            arrow = '⬇️'
        else:
            row_color = '#c2c2c2'
            arrow = '➡️'
        
        change_row.append(f"{arrow} {abs(current_value - last_value)} {abs(diff):.2f}%")
        colors.append(row_color)
    rows_data.append(change_row)

    headers = data.headers if data.headers else None
    table = ax.table(
        cellText=rows_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    col_widths = get_max_text_widths(rows_data, headers, font_size=14)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        cell.set_height(0.2)
        if headers and row == 0:
            cell.set_facecolor('#05c2f5')
            cell.set_text_props(weight='bold', color='#f4f6f7')
        elif row == 1:
            cell.set_text_props(weight='bold', color='#666666')
        elif row == 2:  
            cell.set_text_props(weight='bold', color=colors[col])
                
    top, left, right, bottom = get_table_position(fig, ax, table)
    if title:
        ax.text(0.01, top + 0.1, title, ha='left', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis("off")
        
    plt.subplots_adjust(hspace=0.8)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")


async def generate_brand_attribute(request: TableRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")

    data = request.data
    title = data.title
    rows_data = data.rows

    fig, ax = plt.subplots(figsize=(15, 5))
    headers = data.headers if data.headers else None

    table = ax.table(
        cellText=rows_data,
        colLabels=headers,
        cellLoc='center',  
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    col_widths = get_max_text_widths(rows_data, headers, font_size=10)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        cell.set_height(0.1) 
        if headers and row == 0:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold', color='black')  
        elif col == 0:
            cell.set_text_props(ha='left', weight='bold', color='#666666')           
        elif col == 1:
            cell.set_text_props(ha='right', weight='bold', color='#666666')       
        else:
            cell.set_text_props( color='#666666')
            
    top, left, right, bottom  = get_table_position(fig, ax, table)
    ax.text(0.01, top + 0.1, title, ha='left', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

async def generate_channel_distribution(request: TableRequest):
    if not request.data:
        return StreamingResponse(BytesIO(), media_type="image/png")

    data = request.data
    title = data.title
    rows_data = data.rows

    fig, ax = plt.subplots(figsize=(15, 5))
    headers = data.headers if data.headers else None

    table = ax.table(
        cellText=rows_data,
        colLabels=headers,
        cellLoc='center',  
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    col_widths = get_max_text_widths(rows_data, headers, font_size=10)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        cell.set_height(0.1)  
        text_color = 'black' if (headers and row == 0) else '#666666'
        if col == 0:
            ha = 'left'
        elif col in (1, 2):
            ha = 'right'
        else:
            ha = 'center'
        if headers and row == 0:
            cell.set_facecolor('#f0f0f0')
        cell.set_text_props(ha=ha, weight='bold', color=text_color)
            
    top, left, right, bottom = get_table_position(fig, ax, table)
    ax.text(0.01, top + 0.1, title, ha='left', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")
