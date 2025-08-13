import os
import uuid
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv

# --- 1. 환경 변수 로드 ---
load_dotenv()

# ⭐️ 사용자님 지적대로, 코드 전체에서 사용할 변수 이름을 환경 변수 키와 동일하게 통일합니다.
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# --- 2. 클라이언트 초기화 (오류 방지 로직 포함) ---
blob_service_client = None

# 환경 변수가 모두 존재할 때만 클라이언트 초기화를 시도합니다.
if AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME:
    try:
        # ⭐️ AZURE_STORAGE_CONNECTION_STRING 변수를 사용하여 초기화
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        print("[STORAGE] Azure Blob Storage 서비스 클라이언트 초기화 성공.")
    except Exception as e:
        print(f"[STORAGE] 오류: 스토리지 클라이언트 초기화 실패. 연결 문자열 값을 확인하세요. - {e}")
        # 실패해도 앱이 죽지 않고 blob_service_client는 None으로 유지됩니다.
else:
    # 환경 변수가 없는 경우, 경고만 출력하고 앱은 계속 실행됩니다.
    print("[STORAGE] 경고: 스토리지 연결 환경 변수가 설정되지 않았습니다.")


# --- 3. 업로드 함수 ---
async def upload_image_and_get_url(file_bytes: bytes, original_filename: str) -> str:
    """
    이미지를 업로드하고 URL을 반환합니다.
    """
    if not blob_service_client:
        raise ConnectionError("스토리지 서비스가 초기화되지 않았습니다. Azure Portal의 환경 변수 설정을 확인하세요.")

    try:
        file_extension = os.path.splitext(original_filename)[1]
        blob_name = f"images/{uuid.uuid4()}{file_extension}"
        
        # ⭐️ AZURE_STORAGE_CONTAINER_NAME 변수를 사용하여 컨테이너 지정
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
