import os
import uuid
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv

# --- 1. 환경 변수 로드 및 설정 ---
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

if not AZURE_STORAGE_CONNECTION_STRING or not AZURE_STORAGE_CONTAINER_NAME:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING 또는 AZURE_STORAGE_CONTAINER_NAME 환경 변수를 확인해주세요.")

# --- 2. Blob 서비스 클라이언트 초기화 ---
# 모듈이 처음 로드될 때 딱 한 번만 클라이언트를 생성합니다.
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    print("[STORAGE] Azure Blob Storage 서비스 클라이언트 초기화 성공.")
except Exception as e:
    print(f"[STORAGE] 오류: 스토리지 클라이언트 초기화 실패. 연결 문자열을 확인하세요. - {e}")
    blob_service_client = None


# --- 3. [핵심] 최상위 함수로 변경된 업로드 함수 ---
async def upload_image_and_get_url(file_bytes: bytes, original_filename: str) -> str:
    """
    이미지 바이트 데이터를 Azure Blob Storage에 업로드하고 공개 URL을 반환합니다.
    (클래스 없는 버전)
    """
    if not blob_service_client:
        raise ConnectionError("스토리지 서비스가 초기화되지 않았습니다.")

    try:
        file_extension = os.path.splitext(original_filename)[1]
        blob_name = f"images/{uuid.uuid4()}{file_extension}"
        
        # 모듈 수준의 클라이언트를 직접 사용합니다.
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER_NAME,
            blob=blob_name
        )
        
        await blob_client.upload_blob(file_bytes, overwrite=True)
        
        print(f"[STORAGE] 이미지 업로드 성공. URL: {blob_client.url}")
        return blob_client.url

    except Exception as e:
        print(f"[STORAGE] 오류: 이미지 업로드 중 실패 - {e}")
        raise e