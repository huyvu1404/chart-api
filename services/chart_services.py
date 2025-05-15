import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from io import BytesIO
from wordcloud import WordCloud
from fastapi.responses import StreamingResponse
from models import SanKeyChartRequest, PieChartRequest, BarChartRequest, LineChartRequest, WordCloudRequest

random.seed(40)

async def generate_bar_chart(request: BarChartRequest):
    fig, ax = plt.subplots()
    bottom = np.zeros(len(request.x))
   
    if isinstance(request.y[0], list):
        for i, y in enumerate(request.y):
            if isinstance(y, list):
                ax.bar(request.x, y, bottom=bottom, color=request.colors[i], label=request.labels[i])
                bottom += y
    else:
        ax.bar(request.x, request.y, color=request.colors)
    
    fig.suptitle(request.title, x=0.01, ha='left', fontsize=14, weight='bold')
    ax.set_xlabel(request.xlabel)
    ax.set_ylabel(request.ylabel)
    ax.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  

    return StreamingResponse(buf, media_type="image/png")

async def generate_top_social_posts(request: BarChartRequest):
    fig, ax = plt.subplots()
    categories = request.x
    total_engagements, sentiment_score = request.y
    colors = request.colors

    ax.bar(categories, total_engagements, color=colors[0], label='Total Engagements')
    bars2 = ax.bar(categories, [-x/5 if x > 0 else 0 for x in sentiment_score], color=colors[1], label='Sentiment Score')
    
    for i, bar in enumerate(bars2):
        height = -sentiment_score[i] / 5
        if height < 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height - 1, f'{int(abs(height * 5))}%', ha='center', va='top', color='black')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height - 1, '0%', ha='center', va='top', color='red')
    
    ax.set_ylim(-30, 50)
    ax.axhline(0, color="black", linewidth=0.8)
    grid_yticks = [-20, 0, 20, 40]
    ax.set_yticks(grid_yticks)
    ax.set_yticklabels([str(abs(tick)) if tick >= 0 else None for tick in grid_yticks])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
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
    sizes = [d.count for d in request.data]
    labels = [f"{d.label} ({d.count})\n{d.percentage:.2f}%" for d in request.data]
    colors = [d.color for d in request.data]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(sizes, colors=colors, wedgeprops=dict(width=0.3, edgecolor='white'), startangle=90)

    ax.text(0, 0.1, f"{request.total:,}", fontsize=20, weight="bold", ha='center')
    ax.text(0, -0.1, "Mentions", fontsize=12, color='gray', ha='center')

    legend_labels = [f"{d.percentage:.2f}%  {d.label}  ({d.count})" for d in request.data]
    ax.legend(wedges, legend_labels, title=request.title, loc="center left", bbox_to_anchor=(1, 0.5))

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")


async def generate_wordcloud(request: WordCloudRequest):
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

    fig, ax = plt.subplots()
    if isinstance(request.y[0], list):
        for i, y in enumerate(request.y):
            if isinstance(y, list):
                ax.plot(request.x, y, color=request.colors[i], label=request.labels[i])
    else:
        ax.plot(request.x, request.y, color=request.colors, label=request.labels[0])
    
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
    fig, ax = plt.subplots()

    this_week = request.y[1]
    last_week = request.y[0]

    ax.plot(request.x, this_week, color=request.colors[1], label=request.labels[1], marker='o')
    ax.plot(request.x, last_week, color=request.colors[0], label=request.labels[0], linestyle='--', marker='o')

    total_this_week = sum(this_week)
    total_last_week = sum(last_week)
    if total_last_week > 0:
        percentage_change = ((total_this_week - total_last_week) / total_last_week) * 100
    else:
        percentage_change = 0
   
    if "Negative" in request.title:
        emoji = "😞"
    elif "Positive" in request.title:
        emoji = "😃"

    if total_this_week > total_last_week:
        arrow = "⬆️"
    else:
        arrow = "⬇️"

    fig.suptitle(request.title, x=0.01, ha='left', fontsize=14, weight='bold')
    fig.text(0.1, 0.91, f"{total_this_week} {emoji} {arrow}{abs(percentage_change):.2f}% (compare to last week)", ha='left', va='top', fontsize=12)
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
    sources = request.data["sources"]
    sentiments = request.data["sentiments"]
    topics = request.data["topics"]

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

    def get_random_color():
        return "#%06x" % random.randint(0, 0xFFFFFF)

    node_colors = [sentiment_colors.get(label, get_random_color()) for label in labels]

    def hex_to_rgba(hex_color, alpha):
        rgb = mcolors.to_rgb(hex_color)  # (r, g, b) as floats [0-1]
        return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"

    link_colors = [hex_to_rgba(node_colors[src], 0.2) for src in source_list]         # Nhạt (mặc định)
    link_hovercolors = [hex_to_rgba(node_colors[src], 0.6) for src in source_list]    # Đậm hơn khi hover

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=updated_labels,
            color=node_colors,  # <- Fixed colors applied here
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
        width=1000,    
        height=600,
        title_text=f"<b>{request.title}</b>",  # In đậm
        title_x=0.1, 
        font_size=10
    )

    buf = BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")
