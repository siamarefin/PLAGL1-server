

from fastapi import APIRouter, File, UploadFile, Depends, Query, Body, HTTPException, Request
from core.security import verify_token
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import os
import subprocess
from models.schema import OutlierSchema
from core.consts import BASE_URL
import pandas as pd 
import json
import subprocess
from pydantic import BaseModel


pandas2ri.activate()



router = APIRouter(prefix='/operations', tags=['operation'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')



@router.post('/init')
async def init(count_data: UploadFile = File(...), meta_data: UploadFile = File(...), user_info: dict = Depends(verify_token)):

    try:

        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id'])), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "rds"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "files"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "figures"), exist_ok=True)

        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "files")
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")
        file1_path = os.path.join( FILE_DIR, f"count_data.csv")
        with open(file1_path, "wb") as f:
            f.write(await count_data.read())

        file2_path = os.path.join(FILE_DIR, f"meta_data.csv")
        with open(file2_path, "wb") as f:
            f.write(await meta_data.read())
        
        robjects.r(
        f"""
            source("../req_packages.R")
            count_data <- read.csv("files/count_data.csv", header = TRUE, row.names = 1)
            sample_info <- read.csv("files/meta_data.csv", header = TRUE, row.names = 1)
            saveRDS(count_data, "rds/count_data.rds")
            saveRDS(sample_info, "rds/sample_info.rds")
        """)

        return {"message": "file uploadeded & Processed successfully!" ,"count_data": file1_path, "meta_data": file2_path}   

    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    

    



@router.get('/analyze')
async def analyze(user_info: dict = Depends(verify_token)):
    try:

        # robjects.r['setwd'](R_CODE_DIRECTORY)

        run_r_script("analyze.R", [str(user_info['user_id'])])
    except Exception as e:
        return {"message": "Error in analyzing file", "error": str(e)}
    
    return {
        "message": "Analysis completed successfully!",
        "results": {
            "boxplot_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.png",
            "boxplot_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.png",
            "htree_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.png",
            "htree_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.png",
            "pca_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.png",
            "pca_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.png",
            "tsne_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.png",
            "tsne_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.png",
            "umap_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.png",
            "umap_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.png",

            "boxplot_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.pdf",
            "boxplot_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.pdf",
            "htree_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.pdf",
            "htree_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",
            "pca_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.pdf",
            "pca_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.pdf",
            "tsne_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.pdf",
            "tsne_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.pdf",
            "umap_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.pdf",
            "umap_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.pdf",

            "normalized_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/Normalized_Count_Data.csv",
            "count_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/count_data.csv",
            "meta_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/meta_data.csv"
        }
    }


@router.post('/upload-merge')
async def upload_merge(merged_df_data_normalized_t: UploadFile = File(...), user_info: dict = Depends(verify_token)):
    """
    API endpoint to upload and save a merge file (merged_df_data_normalized_t.csv) directly to the files directory.
    """
    try:
        # Define user-specific directories
        user_id = str(user_info['user_id'])
        files_dir = os.path.join(R_CODE_DIRECTORY, user_id, "files")

        # Ensure the directory exists
        os.makedirs(files_dir, exist_ok=True)

        # Save the uploaded file as merge_file.csv
        file_path = os.path.join(files_dir, "merged_df_data_normalized_t.csv")
        with open(file_path, "wb") as f:
            f.write(await merged_df_data_normalized_t.read())

        # Return the success message with file path
        return {
            "message": "File uploaded successfully!",
            "merged_df_data_normalized_t": file_path
        }

    except Exception as e:
        # Handle unexpected errors
        return {
            "message": "Error in uploading file.",
            "error": str(e)
        }




@router.post('/batch-effect-correction')
async def batch_effect_correction(user_info: dict = Depends(verify_token)):
    """
    API endpoint to perform batch effect correction using an R script.
    """
    try:
        # Define the file paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "merged_df_data_normalized_t.csv")
        output_dir = os.path.join("code", user_id, "files")
        r_script_path = os.path.join("code", "batch_effect_correction.R")

        # Check if input file exists
        if not os.path.exists(input_file):
            return {
                "message": "Input file not found.",
                "error": f"File not found at {input_file}"
            }

        # Run the R script
        command = ["Rscript", r_script_path, input_file, output_dir,user_id]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Capture output and error
        error = result.stderr.strip()

        if result.returncode != 0:
            return {
                "message": "Error running the batch effect correction script.",
                "error": error
            }

        # List all files created in the output directory
        created_files = [
            f"{BASE_URL}/files/{user_id}/{file_name}"
            for file_name in os.listdir(output_dir)
        ]

        return {
            "message": "Batch effect correction completed successfully.",
            "files_created": created_files
        }

    except Exception as e:
        # Handle unexpected errors
        return {
            "message": "An unexpected error occurred.",
            "error": str(e)
        }



from fastapi import APIRouter, Depends
from code.code import process_file
import os


@router.get('/z_score_normalize')
async def z_score_normalize(user_info: dict = Depends(verify_token)):
    """
    API endpoint to normalize data using Z-score normalization.
    """
    try:
        # Define paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join(
            "code", user_id, "files", "batch_merged_df_data_normalized_t.csv"
        )
        output_dir = os.path.join("code", user_id, "files")
        
        # Verify if the input file exists
        if not os.path.exists(input_file):
            return {
                "message": "Input file not found.",
                "error": f"File not found at {input_file}"
            }

        # Run the normalization function
        result = process_file(input_file, output_dir)

        # Check if normalization was successful
        if result.get("message") == "Normalization completed successfully.":
            return {
                "message": result["message"],
                "normalized_file": result["normalized_file"]
            }
        else:
            return {
                "message": "Normalization failed.",
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        # Handle unexpected errors
        return {
            "message": "An unexpected error occurred.",
            "error": str(e)
        }




from code.code import visualize_dimensionality_reduction 

@router.get('/visualize-dimensions')
async def dimensionality_reduction(user_info: dict = Depends(verify_token)):
    """
    API endpoint to perform dimensionality reduction and create visualizations.
    """
    try:
        # Define input file and output directory paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join(
            "code", user_id, "files", "z_score_normalized_data.csv"
        )
        output_dir = os.path.join("code", user_id, "files")

        # Verify if the input file exists
        if not os.path.exists(input_file):
            return {
                "message": "Input file not found.",
                "error": f"File not found at {input_file}"
            }

        # Call the visualization function
        result = visualize_dimensionality_reduction(input_file, output_dir)

        # Check if the visualizations were successfully created
        if result.get("message") == "Dimensionality reduction visualizations created successfully.":
            return {
                "message": result["message"],
                "visualizations": result
            }
        else:
            return {
                "message": "Visualization failed.",
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        # Handle unexpected errors
        return {
            "message": "An unexpected error occurred.",
            "error": str(e)
        }

from code.code import plot_correlation_clustermap
@router.get('/plot-correlation-clustermap')
async def correlation_clustermap(user_info: dict = Depends(verify_token)):
    """
    API endpoint to generate a correlation clustermap.
    """
    try:
        # Define input and output paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "z_score_normalized_data.csv")
        output_dir = os.path.join("code", user_id, "files")
        drop_column = "condition"

        # Verify if the input file exists
        if not os.path.exists(input_file):
            return {
                "message": "Input file not found.",
                "error": f"File not found at {input_file}"
            }

        # Call the `plot_correlation_clustermap` function
        result = plot_correlation_clustermap(input_file, output_dir, drop_column)

        # Return the response from the function
        if result.get("message") == "Correlation clustermap created successfully.":
            return {
                "message": result["message"],
                "output_files": result["output_files"]
            }
        else:
            return {
                "message": "Clustermap generation failed.",
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        # Handle unexpected errors
        return {
            "message": "An unexpected error occurred.",
            "error": str(e)
        }
    

from code.code import feature_selection_and_model
@router.get('/feature-selection-model')
async def feature_selection_model(
    user_info: dict = Depends(verify_token),
    selection_ratio: float = Query(..., ge=0.1, le=1.0, description="Feature selection ratio (0.1 to 1.0)")
):
    """
    API endpoint to perform feature selection and model training with a configurable selection ratio.
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "z_score_normalized_data.csv")
        output_dir = os.path.join("code", user_id, "files")

        # Verify input file exists
        if not os.path.exists(input_file):
            return {
                "message": "Input file not found.",
                "error": f"File not found at {input_file}"
            }

        # Call the feature selection and model training function
        result = feature_selection_and_model(input_file, output_dir, selection_ratio)

        # Parse the result returned by the function
        parsed_result = json.loads(result)

        # Return the parsed response
        if parsed_result.get("message") == "Feature selection and model training completed successfully.":
            return parsed_result
        else:
            return {
                "message": "Feature selection and model training failed.",
                "error": parsed_result.get("error", "Unknown error")
            }

    except Exception as e:
        return {
            "message": "An unexpected error occurred.",
            "error": str(e)
        }



from code.code import benchmark_models

@router.get('/find-best-model')
async def benchmark_models_api(user_info: dict = Depends(verify_token)):
    try:
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "selected_features.csv")
        output_dir = os.path.join("code", user_id, "files")
        os.makedirs(output_dir, exist_ok=True) 
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_file):
            return {"message": "Input file not found.", "error": f"File not found at {input_file}"}

        result = benchmark_models(input_file, output_dir)

        # Ensure result is a dictionary
        if not isinstance(result, dict):
            return {"message": "Unexpected error in model benchmarking.", "error": "Invalid return type from function"}

        # Handle errors returned from benchmark_models
        if "error" in result:
            return {"message": "Model benchmarking failed.", "error": result["error"]}

        return {
            "message": "Model benchmarking completed successfully.",
            "metrics": result["metrics"],
            "metrics_file": result["metrics_path"]
        }

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}








from code.code import get_model_and_importance_with_top10, best_models
from fastapi import Form
global_model_name  = "Extra Trees" 

@router.post('/top10-features')
async def top10_features(model_name: str = Form(...), user_info: dict = Depends(verify_token)):


    """
    API endpoint to extract top 10 features for a selected model.
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        reduced_df_path = os.path.join("code", user_id, "files", "selected_features.csv")
        output_dir = os.path.join("code", user_id, "files")


        global_model_name =  model_name 

        # Load reduced DataFrame
        if not os.path.exists(reduced_df_path):
            return {"message": "Reduced DataFrame file not found.", "error": f"File not found at {reduced_df_path}"}
        reduced_df = pd.read_csv(reduced_df_path)

        # Check if the selected model exists
        if global_model_name not in best_models:
            return {"message": f"Model '{global_model_name}' not found in best models."}

        # Call the function to get top 10 features
        result = get_model_and_importance_with_top10(
            metrics_df=None,  # Metrics are not needed for this function
            best_models=best_models,
            reduced_df=reduced_df,
            selected_model_name=global_model_name,
            output_dir=output_dir
        )

        return {
            "message": "Top 10 features extracted successfully.",
            "top10_features": result["top10_features"],
            "top10_features_file": result["top10_features_path"],
            "top10_plot_file": result["top10_plot_path"]
        }

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}




from code.code import visualize_dimensionality_reduction_feature

@router.get('/visualize_dimensionality_reduction_feature')
async def visualize_dimensions_api(
    user_info: dict = Depends(verify_token)
):
    """
    API endpoint to perform dimensionality reduction visualization (PCA, t-SNE, UMAP).
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "top10_features_Extra Trees.csv")
        output_dir = os.path.join("code", user_id, "files")

        # Ensure the input file exists
        if not os.path.exists(input_file):
            return {"message": "Input file not found.", "error": f"File not found at {input_file}"}

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Call the function to generate visualizations
        result = visualize_dimensionality_reduction_feature(input_file, output_dir)

        # Check for errors in the result
        if "error" in result:
            return {"message": "Visualization failed.", "error": result["error"]}

        return {
            "message": result["message"],
            "combined_visualization": result["Combined"]
        }

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}



from code.code import rank_features, param_grids, classifiers

@router.get('/evaluate-single-features')
async def rank_features_api(
    user_info: dict = Depends(verify_token)
):
    """
    API endpoint to rank features based on individual model performance metrics.
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "top10_features_Extra Trees.csv")
        output_dir = os.path.join("code", user_id, "files")

        # Ensure the input file exists
        if not os.path.exists(input_file):
            return {"message": "Input file not found.", "error": f"File not found at {input_file}"}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Call the feature ranking function
        result = rank_features(input_file, global_model_name, param_grids, classifiers, output_dir)

        # Check for errors in the result
        if "error" in result:
            return {"message": "Feature ranking failed.", "error": result["error"]}

        return {
            "message": result["message"],
            "ranking_file": result["ranking_file"],
            "metrics": result["metrics"]
        }

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}






from code.code import evaluate_model_with_features

@router.post('/evaluate_model_with_features')
async def evaluate_model_features_api(
    user_info: dict = Depends(verify_token)
):
    """
    API endpoint to evaluate models with varying numbers of top features.
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "top10_features_Extra Trees.csv")
        output_dir = os.path.join("code", user_id, "files")

        # Ensure the input file exists
        if not os.path.exists(input_file):
            return {"message": "Input file not found.", "error": f"File not found at {input_file}"}

        # Call the function
        result = evaluate_model_with_features(input_file, global_model_name, param_grids, classifiers, output_dir)

        # Handle errors
        if "error" in result:
            return {"message": "Evaluation failed.", "error": result["error"]}

        return {
            "message": result["message"],
            "metrics_file": result["metrics_file"],
            "plot_png": result["plot_png"],
            "plot_pdf": result["plot_pdf"],
            "metrics": result["metrics"]
        }

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}


from code.code import visualize_dimensionality_reduction_final

@router.get('/visualize_dimensionality_reduction_final')
async def visualize_dimensions_api(
    user_info: dict = Depends(verify_token)
):
    """
    API endpoint to perform dimensionality reduction visualization (PCA, t-SNE, UMAP).
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        input_file = os.path.join("code", user_id, "files", "final_selected_features_auprc.csv")
        output_dir = os.path.join("code", user_id, "files")

        # Ensure the input file exists
        if not os.path.exists(input_file):
            return {"message": "Input file not found.", "error": f"File not found at {input_file}"}

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Call the function to generate visualizations
        result = visualize_dimensionality_reduction_final(input_file, output_dir)

        # Check for errors in the result
        if "error" in result:
            return {"message": "Visualization failed.", "error": result["error"]}

        return {
            "message": result["message"],
            "combined_visualization": result["Combined"]
        }

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}



from code.code import evaluate_final_model

@router.get('/evaluate-final-model')
async def evaluate_final_model_api(
    user_info: dict = Depends(verify_token)
):
    """
    API endpoint to evaluate the final model and save results.
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        final_df_path = os.path.join("code", user_id, "files", "final_selected_features_auprc.csv")
        output_dir = os.path.join("code", user_id, "files")

        # Ensure the input file exists
        if not os.path.exists(final_df_path):
            return {"message": "Input file not found.", "error": f"File not found at {final_df_path}"}

        # Call the function
        result = evaluate_final_model(final_df_path, global_model_name, param_grids, classifiers, output_dir)

        # Handle errors
        if "error" in result:
            return {"message": "Evaluation failed.", "error": result["error"]}

        return result

    except Exception as e:
        return {"message": "An unexpected error occurred.", "error": str(e)}










































@router.get('/getcolumnspy')
async def get_columns_api(user_info: dict = Depends(verify_token)):
    """
    Runs the external Python script (columns_name.py) to fetch column names.
    """
    try:
        # Run the external Python file
        result = subprocess.run(
            ["python", "code/columns_name.py"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Capture the output and error
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            # Parse the JSON output from the script
            response = json.loads(output)  # json module is used here
            return {
                "message": "Column names fetched successfully!",
                "columns": response.get("columns", [])
            }
        else:
            return {
                "message": "Failed to fetch column names.",
                "error": error
            }

    except Exception as e:
        return {
            "message": "Error running the script.",
            "error": str(e)
        }
    

@router.get('/getcolumnsR')
async def get_columns_api(user_info: dict = Depends(verify_token)):
    """
    Runs the external R script (columns_name.R) to fetch column names.
    """
    try:
        # Path to the R script and user_id
        r_script_path = "code/columns_name.R"
        user_id = str(user_info['user_id'])

        # Run the external R script
        result = subprocess.run(
            ["Rscript", r_script_path, user_id],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Capture the output and error
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            # Parse the JSON output from the R script
            response = json.loads(output)
            if response.get("success"):
                return {
                    "message": "Column names fetched successfully!",
                    "columns": response.get("columns", [])
                }
            else:
                return {
                    "message": "Failed to fetch column names.",
                    "error": response.get("message", "Unknown error")
                }
        else:
            return {
                "message": "Failed to fetch column names.",
                "error": error
            }

    except Exception as e:
        return {
            "message": "Error running the script.",
            "error": str(e)
        }






@router.post('/remove_outliers')
async def remove_outliers(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:

        robjects.r['setwd'](R_CODE_DIRECTORY)

        print(f"current python wd: {os.getcwd()}")

        file_path = f"{user_info['user_id']}/rds/genes.rds"
        

        print(robjects.r('getwd()'))

        genes = data.genes
        r_genes_list = robjects.StrVector(genes)
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(r_genes_list, file=file_path)

        # robjects.r['setwd'](R_CODE_DIRECTORY)
        # robjects.r['setwd'](R_CODE_DIRECTORY)
        # print("ekhane ashenai")
        run_r_script("remove_outlier.R", [str(user_info['user_id'])])
        run_r_script("analyze.R", [str(user_info['user_id'])]) 


        return {"message": "Outliers removed successfully!", 
                "results": {
                "boxplot_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.png",
                "boxplot_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.png",
                "htree_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.png",
                "htree_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.png",
                "pca_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.png",
                "pca_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.png",
                "tsne_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.png",
                "tsne_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.png",
                "umap_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.png",
                "umap_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.png",

                "boxplot_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.pdf",
                "boxplot_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.pdf",
                "htree_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.pdf",
                "htree_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",
                "pca_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.pdf",
                "pca_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.pdf",
                "tsne_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.pdf",
                "tsne_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.pdf",
                "umap_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.pdf",
                "umap_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.pdf",

                "normalized_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/Normalized_Count_Data.csv",
                "count_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/count_data.csv",
                "meta_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/meta_data.csv"
            }
        }
    
    except Exception as e:
        return {"message": "Error in removing outliers", "error": str(e)}
    



    
@router.get('/conditions')
async def show_conditions(user_info: dict = Depends(verify_token)):
    try:

        pandas2ri.activate()
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")
        readRDS = robjects.r['readRDS']
        condition_vector = readRDS("rds/condition.rds")
        # Extract factor levels
        factor_levels = robjects.r['levels'](condition_vector)        
        # Convert factor levels to a Python list
        factor_levels_list = list(factor_levels)

        # robjects.r['setwd'](R_CODE_DIRECTORY)
        return {"options": factor_levels_list}

        # run_r_script("conditions.R", [str(user_info['user_id'])])
    except Exception as e:
        return {"message": "Error in showing conditions", "error": str(e)}



@router.get('/select_condition')
async def select_condition(option: str, user_info: dict = Depends(verify_token)):
    try:

        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")

        robjects.r(
        f"""
            library(DESeq2)
            dds <- readRDS("rds/dds.rds")
            ref_level <- \"{option}\"
            dds$Treatment <- relevel(dds$Treatment, ref = ref_level)
            dds <- DESeq(dds)
            coeff_names <- as.data.frame(resultsNames(dds))
            saveRDS(coeff_names, "rds/coeff_names.rds")
            saveRDS(ref_level, "rds/ref_level.rds")
            saveRDS(dds, "rds/dds.rds")
        """)

        pandas2ri.activate()

        readRDS = robjects.r['readRDS']
        coeff_vector = readRDS("rds/coeff_names.rds")

        coeff_vector = list(coeff_vector[0])

        # print(coeff_vector[0])

        coeffs = []

        for coeff in coeff_vector:
            coeffs.append(str(coeff))
    
        return {"options": coeffs}
    
    except Exception as e:
        return {"message": "Error in selecting condition", "error": str(e)}
    


    


# this will let the user to select the condition and it will show the coeffs 

@router.get('/volcano')
async def volcano_plot(coeff: str, user_info: dict = Depends(verify_token)):

    try:

        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")

        robjects.r(
        f"""
            coeff_names <- readRDS("rds/coeff_names.rds")
            X <- \"{coeff}\"
            saveRDS(X, "rds/X.rds")
        """)


        readRDS = robjects.r['readRDS']
        X = readRDS("rds/X.rds")


        run_r_script("volcano.R", [str(user_info['user_id'])])
        # f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",

        return {"message": "Volcano plot generated successfully!",
                "results": {
                    "histogram_img": f"{BASE_URL}/figures/{user_info['user_id']}/histogram_pvalues.png",
                    "histogram_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/histogram_pvalues.pdf",
                    "volcano_img": f"{BASE_URL}/figures/{user_info['user_id']}/volcano_plot.png",
                    "volcano_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/volcano_plot.pdf",
                    "upregulated_genes" : f"{BASE_URL}/files/{user_info['user_id']}/Upregulated_padj_{X[0]}.csv",
                    "downregulated_genes" : f"{BASE_URL}/files/{user_info['user_id']}/Downregulated_padj_{X[0]}.csv",
                    "resLFC" : f"{BASE_URL}/files/{user_info['user_id']}/resLFC_{X[0]}.csv"
                }
               }
    
    except Exception as e:
        return {"message": "Error in generating volcano plot", "error": str(e)}




# user will pass the coeff and it will show the volcano plot

@router.post('/highlighted_volcano')
async def highlighted_volcano(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")
        r_genes_to_highlight = robjects.StrVector(data.genes)
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(r_genes_to_highlight, file="rds/genes_to_highlight.rds")
        run_r_script("highlighted_volcano.R", [str(user_info['user_id'])])

        return {"message": "Highlighted Volcano plot generated successfully!",
                "results":{
                    "highlighted_volcano_img": f"{BASE_URL}/figures/{user_info['user_id']}/highlighted_volcano.png",
                    "highlighted_volcano_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/highlighted_volcano.pdf"
                }
               }

        
    except Exception as e:
        return {"message": "Error in generating highlighted volcano plot", "error": str(e)}
    



@router.get('/list_of_files')
async def list_of_files(user_info: dict = Depends(verify_token)):

    def safe_listdir(directory, prefix=""):
        if os.path.exists(directory):
            return [os.path.join(prefix, f) for f in os.listdir(directory)]
        return []

    try:
        USER_DIR = os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']))

        files = (
            safe_listdir(os.path.join(USER_DIR, "files"), "files/") +
            safe_listdir(os.path.join(USER_DIR, "micro/files"), "micro/files/") +
            safe_listdir(os.path.join(USER_DIR, "annotation/files"), "annotation/files/") + 
            safe_listdir(os.path.join(USER_DIR, "heatmap/files"), "heatmap/files/")
        )

        return {"files": files}    
    except Exception as e:
        return {"message": "Error in listing files", "error": str(e)}


@router.get('/list_of_annotated_files')
async def list_of_annotated_files(user_info: dict = Depends(verify_token)):

    def safe_listdir(directory, prefix=""):
        if os.path.exists(directory):
            return [os.path.join(prefix, f) for f in os.listdir(directory)]
        return []

    try:
        USER_DIR = os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']))

        files = (
            safe_listdir(os.path.join(USER_DIR, "annotation/files"), "annotation/files/")
        )

        return {"files": files}    
    except Exception as e:
        return {"message": "Error in listing files", "error": str(e)}




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


#  String api call code 



# Define the request body schema
class MappingPlottingRequest(BaseModel):
    species_name: str
    gene_symbols: list[str]

@router.post("/mapping")
async def run_mapping_plotting(request: MappingPlottingRequest):
    """
    API to run Workflow_Mapping_Plotting_Input_Genes.R and save results.


    {
    "species_name": "Homo sapiens",
    "gene_symbols": ["ZNF212", "ZNF451", "PLAGL1", "NFAT5", "ICAM5", "RRAD"]
}


    """
    try:
        # Define file paths
        r_script_path = "string/String_Sources_Update_v1.R"
        output_dir = "files"
        os.makedirs(output_dir, exist_ok=True)

        # Convert gene symbols list to JSON string
        gene_symbols_json = json.dumps(request.gene_symbols)

        # Run the R script
        command = ["Rscript", r_script_path, request.species_name, gene_symbols_json, output_dir]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Capture stdout and stderr
        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        # Check for errors
        if result.returncode != 0:
            return {"message": "Error running R script", "error": stderr_output}

        return {
            "message": "Workflow completed successfully!",
            "output_directory": output_dir,
            "stdout": stdout_output
        }

    except Exception as e:
        return {"message": "Unexpected error", "error": str(e)}




# Define the request body schema
class MappingPlottingRequest(BaseModel):
    species_name: str
    gene_symbols: list[str]
    clustering_method1: str
    clustering_method2: str

@router.post("/string")
async def run_mapping_plotting(request: MappingPlottingRequest, user_info: dict = Depends(verify_token)):
    """
    API to run Workflow_Mapping_Plotting_Input_Genes.R and save results.

    Example input:
    {
        "species_name": "Homo sapiens",
        "gene_symbols": ["ZNF212", "ZNF451", "PLAGL1", "NFAT5", "ICAM5", "RRAD"],
        "clustering_method": "fastgreedy"
    }
    """
    try:
        # Define file paths
        user_id = str(user_info['user_id'])
        r_script_path = "string/String_Workflow_Update_v3_Feb19.R"
        output_dir = os.path.join("code", user_id, "files")
        os.makedirs(output_dir, exist_ok=True)

        # Convert gene symbols list to JSON string
        gene_symbols_json = json.dumps(request.gene_symbols)

        # Run the R script with species_name, gene_symbols_json, clustering_method, and output_dir as arguments
        command = [
            "Rscript",
            r_script_path,
            request.species_name,
            gene_symbols_json,
            request.clustering_method1,
            request.clustering_method2,
            output_dir
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Capture stdout and stderr
        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        # Check for errors
        if result.returncode != 0:
            return {"message": "Error running R script", "error": stderr_output}

        return {
            "message": "Workflow completed successfully!",
            "output_directory": output_dir,
            "stdout": stdout_output
        }

    except Exception as e:
        return {"message": "Unexpected error", "error": str(e)}
