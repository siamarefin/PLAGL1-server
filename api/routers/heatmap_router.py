from fastapi import APIRouter, File, UploadFile, Depends, Form
from typing import List, Optional, Tuple
from pydantic import BaseModel
from core.security import verify_token
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import shutil
import os
import subprocess
from models.schema import VennSchema, HeatmapSchema
from core.consts import BASE_URL


pandas2ri.activate()

router = APIRouter(prefix='/operations/heatmap', tags=['heatmap'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')



# 2 outcomes from multiple inputs
# outcome 1: heatmap diagram png
# outcome 2: heatmap csv file

@router.post('/init_pipeline')   
async def heatmap_diagram(data: HeatmapSchema, user_info: dict = Depends(verify_token)):
    try:
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "heatmap"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "heatmap", "rds"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "heatmap", "files"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "heatmap", "figures"), exist_ok=True)

        robjects.r['setwd'](R_CODE_DIRECTORY)
        files = data.file_list
        USER_DIR = os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']))        
        for file in files:
            shutil.copyfile(os.path.join(USER_DIR, file) , os.path.join(USER_DIR, "heatmap", "files", file.split("/")[-1]))

        
        
        gene_list = robjects.StrVector(data.gene_list)
        csv_files = robjects.StrVector(["files/"+file.split("/")[-1] for file in files])
        column_names = robjects.StrVector(data.column_names)
        

        saveRDS = robjects.r['saveRDS']
        saveRDS(gene_list, f"{USER_DIR}/heatmap/rds/gene_list.rds")
        saveRDS(csv_files, f"{USER_DIR}/heatmap/rds/csv_files.rds")
        saveRDS(column_names, f"{USER_DIR}/heatmap/rds/column_names.rds")

        # run_r_script("heatmap_api.R", [str(user_info['user_id'])])


        return {"message": "Files and input processed successfully!"}

    except Exception as e:
        return {"message": "Error in processing files", "error": str(e)}
    


@router.post('/upload_annotation_df')
async def upload_annotation_df(ann_df: UploadFile = File(...), user_info: dict = Depends(verify_token)):
    try:
        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "heatmap", "files")
        file1_path = os.path.join( FILE_DIR, f"Heatmap_annotation.csv")
        with open(file1_path, "wb") as f:
            f.write(await ann_df.read())

        return {"message" : "Annodation file uploaded successfully", "file_path": file1_path}

    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}


@router.delete("/remove_annotation_df")
async def remove_annotation_df(user_info: dict = Depends(verify_token)):
    try:
        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "heatmap", "files")
        file1_path = os.path.join( FILE_DIR, f"Heatmap_annotation.csv")
        os.remove(file1_path)

        return {"message" : "Annotation file removed successfully"}

    except Exception as e:
        return {"message": "Error in removing file", "error": str(e)}


@router.get("/heatmap_diagram")
async def get_heatmap_diagram(user_info: dict = Depends(verify_token)):
    try:
        robjects.r['setwd'](R_CODE_DIRECTORY)
        run_r_script("heatmap_api_pipeline.R", [str(user_info['user_id'])])

        return {"message": "Heatmap diagram created successfully!", 
                "heatmap_diagram": f"{BASE_URL}/figures/heatmap/{user_info['user_id']}/heatmap_output.png", 
                "heatmap_csv": f"{BASE_URL}/files/heatmap/{user_info['user_id']}/heatmap_data.csv"}

    except Exception as e:

        return {"message": "Error in creating heatmap diagram", "error": str(e)}



@router.post("/custom_heatmap_diagram")
async def get_custom_heatmap_diagram(heatmap_data: UploadFile = File(...), user_info: dict = Depends(verify_token)):
    try:

        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "heatmap", "files")
        file1_path = os.path.join( FILE_DIR, f"Heatmap_data.csv")
        with open(file1_path, "wb") as f:
            f.write(await heatmap_data.read())


        robjects.r['setwd'](R_CODE_DIRECTORY)
        run_r_script("heatmap_api_custom.R", [str(user_info['user_id'])])

        return {"message": "Heatmap diagram created successfully!", 
                "heatmap_diagram": f"{BASE_URL}/figures/heatmap/{user_info['user_id']}/heatmap_output.png", 
                "heatmap_csv": f"{BASE_URL}/files/heatmap/{user_info['user_id']}/heatmap_data.csv"}

    except Exception as e:

        return {"message": "Error in creating heatmap diagram", "error": str(e)}


def run_r_script(script_name, args=None):
    cmd = ["Rscript", os.path.join(R_CODE_DIRECTORY, script_name)]
    # print(...args)
    if args:
        cmd.extend(args)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if process.returncode != 0:
        raise Exception(f"R script failed: {stderr.decode()}")

    return stdout.decode()
