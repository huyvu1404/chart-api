from fastapi import APIRouter
from models.chart_models import TableRequest, SanKeyChartRequest, BarChartRequest, PieChartRequest, LineChartRequest, WordCloudRequest
from services.chart_services import *

chart = APIRouter()

@chart.post("/topic-distribution")
async def create_topic_distribution(request: SanKeyChartRequest):
    return await generate_topic_distribution(request)
    
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
    return await generate_channel_breakdown(request)

@chart.post("/sentiment-breakdown")
async def create_sentiment_breakdown(request: PieChartRequest):
    return await generate_sentiment_breakdown(request)

@chart.post("/trend")
async def create_trend_chart(request: LineChartRequest):
    return await generate_trend_chart(request)

@chart.post("/conservation-breakdown")
async def create_conservation_breakdown(request: BarChartRequest):
    return await generate_bar_chart(request)

@chart.post("/top-sources")
async def create_top_sources(request: TableRequest):
    return await generate_top_sources(request)

@chart.post("/overview")
async def create_overview(request: TableRequest):
    return await generate_overview(request)

@chart.post("/brand-attribute")
async def create_brand_attribute(request: TableRequest):
    return await generate_brand_attribute(request)

@chart.post("/channel-distribution")
async def create_channel_distribution(request: TableRequest):
    return await generate_channel_distribution(request)