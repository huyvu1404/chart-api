from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Union

class LabeledData(BaseModel):
    label: str
    count: int
    percentage: float
    color: Optional[str] = None 

class PieChartRequest(BaseModel):
    title: str
    values: List[int]
    labels: List[str]
    colors: Optional[List[str]] = None

class AxisLabels(BaseModel):
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

class BarChartRequest(AxisLabels):
    x: List[str]
    y: Union[List[int], List[List[int]]]
    labels : Optional[List[str]] = None
    colors: Optional[Union[str, List[str]]] = Field(default="blue")

class LineChartRequest(AxisLabels):
    x: Optional[List[str]] = None
    y: Optional[Union[List[int], List[List[int]]]] = None
    labels : Optional[List[str]] = None
    colors: Optional[Union[str, List[str]]] = Field(default="blue")
    


class WordCloudRequest(BaseModel):
    title: str
    data: List[dict] 
    color: Optional[str] = Field(default="blue") 

class SanKeyChartRequest(BaseModel):
    title: str
    data: dict
class TableData(BaseModel):
    title: Optional[str] = None
    column_labels: Optional[List[str]] = None
    total: Optional[int] = None
    net_sentiment_score: Optional[float] = None
    rows: List[List[Union[str, int]]]

class TableRequest(BaseModel):
    data: Optional[Union[List[TableData], TableData]] = Field(default_factory=dict)

    @model_validator(mode="before")
    def normalize_data(cls, values):
        data = values.get("data")
        if isinstance(data, dict) and not data:
            # Nếu data là {}, thì coi như không có
            values["data"] = None
        return values