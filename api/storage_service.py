import os
import uuid
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv

# --- 1. 환경 변수 로드 ---
load_dotenv()
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# --- 2. 클라이언트 초기화 (오류 발생시키지 않기) ---
blob_service_client = None

# 환경 변수가 모두 존재할 때만 클라이언트 초기화를 시도합니다.
if CONNECTION_STRING and CONTAINER_NAME:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        print("[STORAGE] Azure Blob Storage 서비스 클라이언트 초기화 성공.")
    except Exception as e:
        print(f"[STORAGE] 오류: 스토리지 클라이언트 초기화 실패. 연결 문자열 값(Value)을 확인하세요. - {e}")
        # 실패해도 앱이 죽지 않고 blob_service_client는 None으로 유지됩니다.
else:
    # 환경 변수가 아예 없는 경우, 경고만 출력하고 앱은 계속 실행됩니다.
    print("[STORAGE] 경고: 스토리지 연결 환경 변수가 설정되지 않았습니다.")


# --- 3. 업로드 함수 ---
async def upload_image_and_get_url(file_bytes: bytes, original_filename: str) -> str:
    """
    이미지를 업로드하고 URL을 반환합니다.
    """
    # 이제 이 함수가 호출되는 시점에 클라이언트가 제대로 초기화되었는지 확인합니다.
    if not blob_service_client:
        # 이 에러가 바로 Azure 로그에 찍혔던 "스토리지 서비스가 초기화되지 않았습니다"의 원인입니다.
        raise ConnectionError("스토리지 서비스가 초기화되지 않았습니다.")

    try:
        file_extension = os.path.splitext(original_filename)[1]
        blob_name = f"images/{uuid.uuid4()}{file_extension}"
        
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME,
            blob=blob_name
        )
        
        await blob_client.upload_blob(file_bytes, overwrite=True)
        
        return blob_client.url

    except Exception as e:
        print(f"[STORAGE] 오류: 이미지 업로드 중 실패 - {e}")
        raise e