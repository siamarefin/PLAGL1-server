from pydantic import BaseModel, EmailStr, Field

from typing import List, Optional, Tuple

from fastapi import UploadFile, File
class UserCreate(BaseModel):
    email: EmailStr  
    name: str
    password: str



class UserLogin(BaseModel):
    email: EmailStr
    password: str
    

class OutlierSchema(BaseModel):
    genes: list[str]
    


class AnnotationSchema(BaseModel):
    organism_name: str
    id_type: str


class AnnotatedVolcanoSchema(BaseModel):
    gene_list: list[str]


class VenFile(BaseModel):
    file_path: str
    name: str

class VennSchema(BaseModel):
    file_list: List[VenFile]
    ven_reg: str

class HeatmapSchema(BaseModel):
    gene_list: List[str]
    file_list: List[str]
    column_names: List[str]