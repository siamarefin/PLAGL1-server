

from fastapi import APIRouter, File, UploadFile, Depends
from core.security import verify_token
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import os
import subprocess
from models.schema import OutlierSchema
from core.consts import BASE_URL


pandas2ri.activate()

router = APIRouter(prefix='/operations/micro', tags=['micro_array'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')



@router.post('/init')
async def init(count_data: UploadFile = File(...), meta_data: UploadFile = File(...), user_info: dict = Depends(verify_token)):

    try:

        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "micro"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "micro", "rds"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "micro", "files"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "micro", "figures"), exist_ok=True)

        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "micro" ,"files")
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}/micro")
        file1_path = os.path.join( FILE_DIR, f"count_data.csv")
        with open(file1_path, "wb") as f:
            f.write(await count_data.read())

        file2_path = os.path.join(FILE_DIR, f"meta_data.csv")
        with open(file2_path, "wb") as f:
            f.write(await meta_data.read())
        
        robjects.r(
        f"""
            # load and install libraries

            source("../../micro_functions.R")

            print(getwd())

            load_and_install_libraries()

            data_files <- load_and_preprocess_data("files/count_data.csv", "files/meta_data.csv")
            count_data_subset <- data_files$count_data_subset
            sample_info <- data_files$sample_info

            count_data_subset_cc <- complete_cases_fx(count_data_subset)
            count_data_normalized <- normalize_data(count_data_subset_cc)


            saveRDS(sample_info, "rds/sample_info.rds")
            saveRDS(count_data_subset, "rds/count_data_subset.rds")
            saveRDS(count_data_subset_cc, "rds/count_data_subset_cc.rds")
            saveRDS(count_data_normalized, "rds/count_data_normalized.rds")
            
        """)

        return {"message": "file uploadeded & Processed successfully!" ,"count_data": file1_path, "meta_data": file2_path}   

    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    

@router.get('/analyze')
async def analyze(user_info: dict = Depends(verify_token)):
    try:

        robjects.r['setwd'](R_CODE_DIRECTORY)

        run_r_script("analyze_micro.R", [str(user_info['user_id'])])
    except Exception as e:
        return {"message": "Error in analyzing file", "error": str(e)}
    
    return {
        "message": "Analysis completed successfully!",
        "results": {
            "boxplot_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Boxplot (Before Normalization).png",
            "boxplot_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Boxplot (After Normalization).png",
            "kmeans_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/K-Means Clustering (Before Normalization).png",
            "kmeans_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/K-Means Clustering (After Normalization).png",

            "pca_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/PCA Plot (Before Normalization).png",
            "pca_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/PCA Plot (After Normalization).png",

            "htree_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Phylogenetic Tree (Before Normalization).png",
            "htree_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Phylogenetic Tree (After Normalization).png",
            
            "tsne_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/t-SNE Plot (Before Normalization).png",
            "tsne_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/t-SNE Plot (After Normalization).png",
            "umap_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/UMAP Plot (Before Normalization).png",
            "umap_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/UMAP Plot (After Normalization).png",

            "count_data_csv": f"{BASE_URL}/files/micro/{user_info['user_id']}/count_data.csv",
            "meta_data_csv": f"{BASE_URL}/files/micro/{user_info['user_id']}/meta_data.csv"
        }
    }




@router.post('/remove_outliers')
async def remove_outliers(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:

        robjects.r['setwd'](R_CODE_DIRECTORY)

        print(f"current python wd: {os.getcwd()}")

        file_path = f"{user_info['user_id']}/micro/rds/outliers.rds"
        

        print(robjects.r('getwd()'))

        genes = data.genes
        r_genes_list = robjects.StrVector(genes)
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(r_genes_list, file=file_path)

        # robjects.r['setwd'](R_CODE_DIRECTORY)
        # robjects.r['setwd'](R_CODE_DIRECTORY)
        print("ekhane ashenai")
        run_r_script("remove_outlier_micro.R", [str(user_info['user_id'])])
        return {
            "message": "Outliers removed successfully!", 
            "results": {
                "boxplot_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Boxplot (Before Normalization).png",
                "boxplot_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Boxplot (After Normalization).png",
                "kmeans_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/K-Means Clustering (Before Normalization).png",
                "kmeans_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/K-Means Clustering (After Normalization).png",

                "pca_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/PCA Plot (Before Normalization).png",
                "pca_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/PCA Plot (After Normalization).png",

                "htree_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Phylogenetic Tree (Before Normalization).png",
                "htree_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/Phylogenetic Tree (After Normalization).png",
                
                "tsne_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/t-SNE Plot (Before Normalization).png",
                "tsne_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/t-SNE Plot (After Normalization).png",
                "umap_denorm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/UMAP Plot (Before Normalization).png",
                "umap_norm_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/UMAP Plot (After Normalization).png",

                "count_data_csv": f"{BASE_URL}/files/micro/{user_info['user_id']}/count_data.csv",
                "meta_data_csv": f"{BASE_URL}/files/micro/{user_info['user_id']}/meta_data.csv"
            }
        }
    
    except Exception as e:
        return {"message": "Error in removing outliers", "error": str(e)}
    



    
@router.get('/conditions')
async def show_conditions(user_info: dict = Depends(verify_token)):
    try:
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}/micro")
        readRDS = robjects.r['readRDS']
        sample_info_clean = readRDS("rds/sample_info_clean.rds")
        group = sample_info_clean.rx(True, 1)
        conditions = list(set(group))
        return {"options": conditions}
    except Exception as e:
        return {"message": "Error in showing conditions", "error": str(e)}




@router.get('/volcano')
async def volcano_plot(reference: str, user_info: dict = Depends(verify_token)):

    try:

        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}/micro")

        robjects.r(
        f"""
            Reference <- "{reference}"
            saveRDS(Reference, "rds/Reference.rds")
        """)

        print("etotuku no problem")

        run_r_script("volcano_micro.R", [str(user_info['user_id'])])


        readRDS = robjects.r['readRDS']
        treat = readRDS("rds/treat.rds")

        # run_r_script("volcano.R", [str(user_info['user_id'])])
        # f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",

        return {"message": "Volcano plot generated successfully!",
                "results": {
                    "volcano_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/volcano_plot.png",
                    "upregulated_genes" : f"{BASE_URL}/files/micro/{user_info['user_id']}/Upregulated_genes_{treat[0]}_vs_{reference}.csv",
                    "downregulated_genes" : f"{BASE_URL}/files/micro/{user_info['user_id']}/Downregulated_genes_{treat[0]}_vs_{reference}.csv",
                    "resLFC" : f"{BASE_URL}/files/micro/{user_info['user_id']}/LFC.csv"
                }
               }
    
    except Exception as e:
        return {"message": "Error in generating volcano plot", "error": str(e)}


@router.post('/highlighted_volcano')
async def highlighted_volcano(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}/micro")
        r_genes_to_highlight = robjects.StrVector(data.genes)
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(r_genes_to_highlight, file="rds/highlight_genes.rds")


        run_r_script("highlighted_volcano_mirco.R", [str(user_info['user_id'])])

        return {"message": "Highlighted Volcano plot generated successfully!",
                "results":{
                    "highlighted_volcano_img": f"{BASE_URL}/figures/micro/{user_info['user_id']}/volcano_plot_highlighted.png",
                    # "highlighted_volcano_pdf": f"{BASE_URL}/figures/micro/{user_info['user_id']}/highlighted_volcano.pdf"
                }
               }

        
    except Exception as e:
        return {"message": "Error in generating highlighted volcano plot", "error": str(e)}
   


# @router.get('/select_condition/{option}')
# async def select_condition(option: int, user_info: dict = Depends(verify_token)):
#     try:

#         robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")

#         robjects.r(
#         f"""
#             library(DESeq2)
#             condition <- readRDS("rds/condition.rds")
#             dds <- readRDS("rds/dds.rds")
#             condition <- as.data.frame(condition)
#             ref_level <- as.character(condition[{option},]) 
#             dds$Treatment <- relevel(dds$Treatment, ref = ref_level)
#             dds <- DESeq(dds)
#             coeff_names <- as.data.frame(resultsNames(dds))
#             saveRDS(coeff_names, "rds/coeff_names.rds")
#             saveRDS(ref_level, "rds/ref_level.rds")
#             saveRDS(dds, "rds/dds.rds")
#         """)

#         pandas2ri.activate()

#         readRDS = robjects.r['readRDS']
#         coeff_vector = readRDS("rds/coeff_names.rds")

#         coeff_vector = list(coeff_vector[0])

#         # print(coeff_vector[0])

#         coeffs = []

#         for coeff in coeff_vector:
#             coeffs.append(str(coeff))
    
#         return {"options": coeffs}
    
#     except Exception as e:
#         return {"message": "Error in selecting condition", "error": str(e)}
    


    







# user will pass the coeff and it will show the volcano plot

 






    






@router.get("/test")
async def test():
    # os.makedirs(FILE_DIR, exist_ok=True)
    robjects.r['setwd'](R_CODE_DIRECTORY + "/1")
    ot = robjects.r(
        f"""
            md <- readRDS("rds/count_data.rds")
            head(md)
        """
    )

    robjects.r['setwd'](R_CODE_DIRECTORY)

    print(ot)

    return {"message": "Tested successfully!"}
    # run_r_script("test.R")



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