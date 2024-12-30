## Running the server

1. Create a virtual environment
   for linux and mac

```bash
python3 -m venv venv
```

for windows

```bash
python -m venv venv
```

2. Activate the virtual environment
   for linux and mac

```bash
source venv/bin/activate
```

for windows

```bash
venv\Scripts\activate
```

3. Go to api directory and Install the dependencies

```bash
cd api
```

```bash
pip install -r requirements.txt
```

4. Run the server

```bash
fastapi dev main.py
```

## Testing the server

1. Install [Postman](https://www.postman.com/downloads/)

2. Open postman and import the collection from the file `postman_collection.json`

3. Run the collection

4. You can test the endpoints by running the requests in the collection

# R package install by siam 
# Step 1: Check if BiocManager is installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Step 2: Install sva from Bioconductor
tryCatch({
  BiocManager::install("sva")
  library(sva)
  cat("sva package installed and loaded successfully!\n")
}, error = function(e) {
  cat("Error installing sva package: ", e$message, "\n")
})

# Step 3: Check Bioconductor validity (optional)
tryCatch({
  BiocManager::valid()
}, error = function(e) {
  cat("Error validating Bioconductor: ", e$message, "\n")
})

# Step 4: If sva installation fails, try an older version
tryCatch({
  packageurl <- "https://bioconductor.org/packages/3.14/bioc/src/contrib/sva_3.38.0.tar.gz"
  install.packages(packageurl, repos = NULL, type = "source")
  library(sva)
  cat("sva package installed from source successfully!\n")
}, error = function(e) {
  cat("Error installing sva package from source: ", e$message, "\n")
})



# python install by siam 
pip install umap-learn 


