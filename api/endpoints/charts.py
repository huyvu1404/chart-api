from fastapi import APIRouter
from models.chart_models import SanKeyChartRequest, BarChartRequest, PieChartRequest, LineChartRequest, WordCloudRequest
from services.chart_services import *

chart = APIRouter()

@chart.post("/sankey-chart")
async def create_sankey_chart(request: SanKeyChartRequest):
    return await generate_sankey_chart(request)
    
@chart.post("/sov")
async def create_sov_chart(request: BarChartRequest):
    return await generate_bar_chart(request)

@chart.post("/top-social-posts")
async def create_top_social_posts(request: BarChartRequest):
    return await generate_top_social_posts(request)

@chart.post("/wordcloud")
async def create_wordcloud(request: WordCloudRequest):
    return await generate_wordcloud(request)

@chart.post("/channel-breakdown")
async def create_channel_breakdown(request: PieChartRequest):
    return await generate_pie_chart(request)

@chart.post("/sentiment-breakdown")
async def create_sentiment_breakdown(request: PieChartRequest):
    return await generate_pie_chart(request)

@chart.post("/trend")
async def create_trend_chart(request: LineChartRequest):
    return await generate_line_chart(request)

@chart.post("/conservation-breakdown")
async def create_conservation_breakdown(request: BarChartRequest):
    return await generate_bar_chart(request)

