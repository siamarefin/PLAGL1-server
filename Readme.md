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
