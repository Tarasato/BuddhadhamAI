.env
```
DATABASE_URL = "sqlserver://127.0.0.1:1433;initial catalog=DB_NAME;user=Username;password=P@assw0rd;trustServerCertificate=true;charset=utf8mb4"

conn_str = "mssql+pyodbc://Username:P@assw0rd@127.0.0.1:1433/DB_NAME?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"

DB_SERVER = "127.0.0.1"
DB_PORT = "0000"
DB_USER = "Username"
DB_PASSWORD = "P4ssw0rd"
DB_NAME = "DB_NAME"
DB_DRIVER = "ODBC Driver 18 for SQL Server"

API_SERVER = "127.0.0.1"
API_SERVER_PORT = 3000

AI_SERVER = "127.0.0.1"
AI_SERVER_PORT = 0000

DEBUG = false
DEBUG_TIME = 5
```

```
to ai
{
    "args": ["ทุกข์", "-k 7", "-d 8"]
}

to api
{
    "question": "มรรค",
    "-k": 7,
    "-d": 100
}
```

sqlcmd -S <ServerIP> -U <Username> -P <Password>
  
output   = "../generated/prisma" If you have ยพรหทฟ problems, remove this from schema.prisma
npx prisma migrate reset --force
npx prisma migrate dev --name init

git rm --cached config.json

python buddhamAI_cli.py ""

pip install -r requirements.txt

pip install faiss-cpu
pip install numpy
pip install ollama

#to do list
docker build -t buddham_ai .

docker-compose up

docker build -t buddham_ai . && docker-compose up -d