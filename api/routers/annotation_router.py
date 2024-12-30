

from fastapi import APIRouter, File, UploadFile, Depends, Form
from typing import List, Optional
from pydantic import BaseModel
from core.security import verify_token
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import os
import subprocess
from models.schema import OutlierSchema, AnnotatedVolcanoSchema
from core.consts import BASE_URL


pandas2ri.activate()

router = APIRouter(prefix='/operations/annotation', tags=['annotation'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')




@router.get('/organism_names')
async def init(user_info: dict = Depends(verify_token)):

    try:

        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "annotation"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "annotation", "rds"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "annotation", "files"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "annotation", "figures"), exist_ok=True)
        


        robjects.r['setwd'](R_CODE_DIRECTORY)

        run_r_script("organism_names.R", [str(user_info['user_id'])])

        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}/annotation")
        

        readRDS = robjects.r['readRDS']
        organisms = readRDS(f"rds/Organisms.rds")

        organisms = list(organisms)

        return {"message": "organism names fetched successfully!", "organisms": organisms}   

    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    

@router.post('/annotate_genes')
async def annotate_genes(files: List[UploadFile] = File(None), organism_name: str = Form(...), id_type: str = Form(...) , user_info: dict = Depends(verify_token)):

    try:
        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "annotation" ,"files")
        file_names = [file.filename for file in files] if files else []
        for file in files:
            with open(os.path.join(FILE_DIR, file.filename), "wb") as f:
                f.write(file.file.read())

        robjects.r['setwd'](R_CODE_DIRECTORY)


        csv_file_paths = robjects.StrVector(file_names)
        path = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "annotation" ,"rds")
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(csv_file_paths, file=path + "/csv_file_paths.rds")




        # make organism_name and id_type R objects using robjects
        robjects.r(
        f"""
            organism_name <- "{organism_name}"
            id_type <- "{id_type}"
            saveRDS(organism_name, "{user_info['user_id']}/annotation/rds/organism_name.rds")
            saveRDS(id_type, "{user_info['user_id']}/annotation/rds/id_type.rds")
        """
        )


        run_r_script("annotate_genes.R", [str(user_info['user_id']), organism_name, id_type])

        # "meta_data_csv": f"{BASE_URL}/files/micro/{user_info['user_id']}/meta_data.csv"

        urls = {}
        for file_name in file_names:
            urls[file_name] = f"{BASE_URL}/files/annotation/{user_info['user_id']}/annotated_{file_name}"
        
        return {"message": "Genes annotated successfully!", "annotated_csv": urls}
    
    except Exception as e:

        return {"message": "Error in annotating genes", "error": str(e)}
    


@router.post('/annotated_volcano')
async def annotated_volcano(data: AnnotatedVolcanoSchema, user_info: dict = Depends(verify_token)):

    try:
        robjects.r['setwd'](R_CODE_DIRECTORY)


        file_path = f"{user_info['user_id']}/annotation/rds/gene_ids.rds"
    

        genes = data.gene_list
        r_genes_list = robjects.StrVector(genes)
        r_saveRDS = robjects.r['saveRDS']
        r_readRDS = robjects.r['readRDS']   
        r_saveRDS(r_genes_list, file=file_path)

        run_r_script("annotated_volcano.R", [str(user_info['user_id'])])

        
        output_file_paths = r_readRDS(f"{user_info['user_id']}/annotation/rds/output_file_paths.rds")

        print(output_file_paths)
        # remove files/ from the file paths
        output_file_paths = [file_path.split("files/")[1] for file_path in output_file_paths]

        volcano_urls = {}

        for file_path in output_file_paths:
            volcano_urls[file_path] = f"{BASE_URL}/figures/annotation/{user_info['user_id']}/Volcano_{file_path}.png"


# annotated_Downregulated_padj_7_LFC_Mice_FROM_16_Samples
        file_urls = {}
        for file_path in output_file_paths:
            # remove annotated_ from the file path

            file_path = file_path.replace("annotated_", "")
            
            file_urls["downregulated_"+file_path] = f"{BASE_URL}/files/annotation/{user_info['user_id']}/annotated_Downregulated_padj_{file_path}"
            file_urls["upregulated_"+file_path] = f"{BASE_URL}/files/annotation/{user_info['user_id']}/annotated_Upregulated_padj_{file_path}"

            
        return {"message": "Volcano plot generated successfully!", "volcano": volcano_urls, "files": file_urls}

    except Exception as e:
        return {"message": "Error in generating volcano plot", "error": str(e)}







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




# print(r_script_dir)

# robjects.r['setwd'](r_script_dir)

# robjects.r(""" 
# l <- c(1,2,3,4,5)
# saveRDS(l, "l.rds")
# """
# )


# def fun():

#     print(os.getcwd())

#     try:


#         if os.path.exists('test.R'):
#             robjects.r('source("test.R")')
#         else:
#             print("File test.R not found.")

#     except Exception as e:
#         print(e)


# def test():
#     fun()

# test()

# def best():
#     robjects.r("source('test2.R')")


# best()