## 필요한 모듈 import

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import csv
import logging

import torch
from typing import Tuple, List, Sequence, Callable, Dict
from pathlib import Path
import sqlite3
import bcrypt
from pydantic import BaseModel

from detectron2.structures import BoxMode
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Depends, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import RedirectResponse

import subprocess

# FastAPI 애플리케이션 인스턴스를 생성
app = FastAPI()
security = HTTPBasic()

# 디렉터리 및 파일 경로
UPLOAD_DIR = "./test_imgs"
OUTPUT_DIR = "./out"
column_DIR = './columns/HKAA.csv'

logging.basicConfig(level=logging.DEBUG)

# 디렉터리 존재 여부 확인
for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# FastAPI용 정적 파일 경로 마운트
app.mount("/static", StaticFiles(directory="../hkaa/static"), name="static")
app.mount("/images", StaticFiles(directory="../hkaa/images"), name="images")
app.mount("/test_imgs", StaticFiles(directory=UPLOAD_DIR), name="test_imgs")
app.mount("/out", StaticFiles(directory=OUTPUT_DIR), name="out")

# Jinja2 템플릿 디렉토리
templates = Jinja2Templates(directory="../hkaa/templates")


# 업로드 및 변환 함수
@app.post("/upload_and_analyze/")
async def upload_and_analyze(request: Request, image: UploadFile = File(...)):
    # 업로드한 이미지를 특정 위치에 저장
    print(image.filename)
    uploaded_image_path = os.path.join(UPLOAD_DIR, image.filename)
    image_name = os.path.splitext(uploaded_image_path)
    image_name = os.path.basename(image_name[0])
    with open(uploaded_image_path, "wb") as buffer:
        buffer.write(image.file.read())

    # 분석 스크립트를 사용하여 이미지 분석
    result = subprocess.run(["python", "HKAA_inference_web.py"], capture_output=True, text=True)

    # HKAA_inference_web.py 스크립트가 'HKAA_result.csv'라는 이름의 파일을 출력으로 생성한다고 가정
    csv_path = os.path.join(OUTPUT_DIR, f'HKAA_result_{image_name}.csv')

    with open(csv_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        data = [row for row in reader]

    analyzed_image_name = "result_" + image.filename
    return {"data": data, "image_path": f"/out/{analyzed_image_name}"}

@app.get("/delete_folders_content/{selected_image_name}")
async def delete_folders_content(selected_image_name: str):
    folders = [UPLOAD_DIR, OUTPUT_DIR]
    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists() and folder_path.is_dir():
            for file in folder_path.iterdir():
                if file.name != selected_image_name:  # 선택한 이미지를 제외하고 삭제
                    try:
                        file.unlink()
                    except Exception as e:
                        logging.error(f"Error deleting {file.name}: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"Error deleting {file.name}: {str(e)}")
    return {"message": "Folders content deleted successfully"}




# page (html)

# Database setup
conn = sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, organization TEXT, email TEXT, is_approved BOOLEAN)")
conn.commit()
conn.close()


@app.get("/", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password, is_approved FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "login_failed"})

    if not bcrypt.checkpw(password.encode('utf-8'), user[0]):
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "login_failed"})

    if not user[1]:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "not_approved"})

    response.set_cookie(key="username", value=username)
    return JSONResponse(content={"detail": "login_successful"})

def check_login(request: Request):
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/signup/")
async def serve_index(request: Request):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_directory, "..", "hkaa", "templates", "signup.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.post("/signup/")
async def signup(request: Request, username: str = Form(...), password: str = Form(...), organization: str = Form(...), email: str = Form(...)):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # 이미 존재하는 사용자 이름인지 확인
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    if user:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "사용자 이름이 이미 존재합니다!"})

    # 비밀번호 암호화
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # 사용자 등록
    cursor.execute("INSERT INTO users (username, password, organization, email, is_approved) VALUES (?, ?, ?, ?, ?)", (username, hashed_password, organization, email, False))
    conn.commit()
    conn.close()

    return RedirectResponse(url="/", status_code=303)


@app.get("/explan/")
async def serve_index():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_directory, "..", "hkaa", "templates", "explanation.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/main")
async def serve_index():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_directory, "..", "hkaa", "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/admin/")
def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.post("/admin/")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    # DB에서 관리자 정보 확인
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM admins WHERE username=?", (username,))
    stored_password = cursor.fetchone()
    conn.close()

    # 비밀번호 일치 확인
    if not stored_password or not bcrypt.checkpw(password.encode('utf-8'), stored_password[0]):
        return JSONResponse(status_code=400, content={"detail": "login_failed"})
    
    # 관리자 대시보드로 리다이렉트
    return JSONResponse(content={"detail": "login_successful"})

# 관리자 대시보드 페이지 라우트
@app.get("/dashboard/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    # 데이터베이스 연결
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # 모든 사용자 정보를 조회
    cursor.execute("SELECT id, username, organization, email, is_approved FROM users")
    raw_users = cursor.fetchall()

    # raw_users에서 가져온 튜플들을 사전 형태로 변환
    users = [{"id": user[0], "username": user[1], "organization": user[2], "email": user[3], "is_approved": user[4]} for user in raw_users]


    # 관리자 대시보드 템플릿을 반환하며 사용자 정보를 전달
    return templates.TemplateResponse("admin_dashboard.html", {"request": request, "users": users})


@app.post("/dashboard/", response_class=HTMLResponse)
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    # TODO: 사용자 이름과 비밀번호 검증 로직 추가
    # 예제로는 'admin'과 'adminpass'를 관리자 로그인 정보로 사용
    if username == "admin" and password == "adminpass":
        # 관리자 대시보드 페이지 반환
        return await admin_dashboard(request)
    else:
        # 오류 메시지와 함께 로그인 페이지 반환
        return templates.TemplateResponse("admin_login.html", {"request": request, "error": "Invalid login credentials"})

# 사용자 승인 기능 라우트
@app.get("/approve/{user_id}")
async def approve_user(user_id: int):
    # 데이터베이스 연결
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # 해당 ID의 사용자를 승인
    cursor.execute("UPDATE users SET is_approved = ? WHERE id = ?", (True, user_id))
    conn.commit()
    conn.close()
    # 승인 후 다시 대시보드 페이지로 리디렉션
    return RedirectResponse(url="/dashboard/")

# 사용자 삭제(탈퇴) 기능 라우트
@app.get("/delete/{user_id}")
async def delete_user(user_id: int):
    # 데이터베이스 연결
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # 해당 ID의 사용자 정보를 삭제
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    # 삭제 후 다시 대시보드 페이지로 리디렉션
    return RedirectResponse(url="/dashboard/")



# 데이터 베이스 보기
@app.get("/view-database/", response_class=HTMLResponse)
async def view_database(request: Request):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # admins 테이블 내용 가져오기
    cursor.execute("SELECT * FROM admins")
    admins = cursor.fetchall()

    # users 테이블 내용 가져오기
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()

    conn.close()

    return templates.TemplateResponse("view_database.html", {"request": request, "admins": admins, "users": users})


@app.put("/admin/reject_user/")
async def reject_user(email: str):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET status='rejected' WHERE email=?", (email,))
    conn.commit()
    conn.close()
    return {"message": f"User {email} rejected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

