FROM python:latest

RUN apt-get update && apt-get install -y \
    r-base \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*


# Install BiocManager
RUN R -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')"

# RUN R -e "install.packages('BiocManager', repos='http://cran.rstudio.com/')"
RUN R -e "BiocManager::install('limma')"

# Install required CRAN packages
RUN R -e "install.packages(c('readr', 'umap', 'ggplot2', 'Rtsne', 'ape'), repos='http://cran.us.r-project.org')"

# Install Bioconductor packages using BiocManager
RUN R -e "BiocManager::install(c('apeglm', 'impute', 'DESeq2', 'WGCNA'))"

# Install CRAN packages
RUN R -e "install.packages(c('here', 'umap', 'Rtsne'), repos='http://cran.us.r-project.org')"

RUN R -e "install.packages(c('Matrix', 'fastcluster', 'dynamicTreeCut', 'flashClust'), repos='http://cran.us.r-project.org')"



WORKDIR /code

COPY ./api/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./api /code/api


EXPOSE 8000

CMD ["fastapi", "dev", "api/main.py", "--host", "0.0.0.0", "--port", "8000"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--proxy-headers"]