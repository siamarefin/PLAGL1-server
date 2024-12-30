from fastapi import APIRouter, File, UploadFile, Depends, Form
from typing import List, Optional, Tuple
from pydantic import BaseModel
from core.security import verify_token
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import shutil
import os
import subprocess
from models.schema import VennSchema
from core.consts import BASE_URL


pandas2ri.activate()

router = APIRouter(prefix='/operations/venn', tags=['venn'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')



# 2 outcomes from multiple csv files
# outcome 1: venn diagram png 
# outcome 2: venn diagram csv file

@router.post('/venn_diagram')   
async def venn_diagram(data: VennSchema , user_info: dict = Depends(verify_token)):
    try:
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "venn"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "venn", "rds"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "venn", "files"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "venn", "figures"), exist_ok=True)

        robjects.r['setwd'](R_CODE_DIRECTORY)
        files = data.file_list
        USER_DIR = os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']))        
        for file in files:
            shutil.copyfile(os.path.join(USER_DIR, file.file_path) , os.path.join(USER_DIR, "venn", "files", file.file_path.split("/")[-1]))

        

        file_paths = robjects.ListVector({"file_paths": robjects.StrVector(["files/"+file.file_path.split("/")[-1] for file in files])})
        gene_list_names = robjects.StrVector([file.name for file in files])

        saveRDS = robjects.r['saveRDS']
        saveRDS(file_paths, f"{USER_DIR}/venn/rds/file_paths.rds")
        saveRDS(gene_list_names, f"{USER_DIR}/venn/rds/gene_list_names.rds")

        

        robjects.r(
              f""" 
                file_paths <- readRDS("{USER_DIR}/venn/rds/file_paths.rds")
                gene_list_names <- readRDS("{USER_DIR}/venn/rds/gene_list_names.rds")
                reg <- "{data.ven_reg}"

                # Ensure file_paths and gene_list_names are vectors, not lists of lists
                if (is.list(file_paths)) {{
                    file_paths <- unlist(file_paths)
                }}
                if (is.list(gene_list_names)) {{
                    gene_list_names <- unlist(gene_list_names)
                }}

                input_data <- vector("list", length(file_paths))  # Initialize the list

                for (i in seq_along(file_paths)) {{
                    input_data[[i]] <- list(
                        file_path = file_paths[i],
                        gene_list_name = gene_list_names[i]
                    )
                }}

                saveRDS(input_data, "{USER_DIR}/venn/rds/input_data.rds")
                saveRDS(reg, "{USER_DIR}/venn/rds/reg.rds")
                """
        )

        run_r_script("venn_diagram.R", [str(user_info['user_id'])])


        return {"message": "Venn diagram created successfully!", "venn_diagram": f"{BASE_URL}/figures/venn/{user_info['user_id']}/venn_diagram_{data.ven_reg}.png", "venn_csv": f"{BASE_URL}/files/venn/{user_info['user_id']}/{data.ven_reg}_venn_result.csv"}

    except Exception as e:
        return {"message": "Error in creating Venn diagram", "error": str(e)}
    


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
