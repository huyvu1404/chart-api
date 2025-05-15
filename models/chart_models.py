from pydantic import BaseModel, Field
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
    title: str
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

class BarChartRequest(AxisLabels):
    x: List[str]
    y: Union[List[int], List[List[int]]]
    labels : Optional[List[str]] = None
    colors: Optional[Union[str, List[str]]] = Field(default="blue")

class LineChartRequest(AxisLabels):
    x: List[str]
    y: Union[List[int], List[List[int]]]
    labels : Optional[List[str]] = None
    colors: Optional[Union[str, List[str]]] = Field(default="blue")

class WordCloudRequest(BaseModel):
    title: str
    data: List[dict] 
    color: Optional[str] = Field(default="blue") 

class SanKeyChartRequest(BaseModel):
    title: str
    data: dict
