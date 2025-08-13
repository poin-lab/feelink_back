import os
import uuid
import logging
import asyncio
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv

# --- 기본 설정 ---
log = logging.getLogger("uvicorn")
load_dotenv()

# --- 환경 변수 로드 ---
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# --- 전역 변수 ---
blob_service_client: BlobServiceClient = None
_init_lock = asyncio.Lock()

async def _initialize_client():
    """
    [내부 함수] BlobServiceClient 인스턴스만 생성합니다.
    (DB의 engine 생성과 유사하게, 이 단계에서 네트워크 연결을 확인하지 않습니다.)
    """
    global blob_service_client
    if not AZURE_STORAGE_CONNECTION_STRING or not AZURE_STORAGE_CONTAINER_NAME:
        log.warning("[STORAGE] 경고: 스토리지 연결 환경 변수가 설정되지 않았습니다.")
        return

    try:
        log.info("[STORAGE] Azure Storage 클라이언트 인스턴스를 생성합니다. (연결 확인은 아직 안 함)")
        # ⭐️ 핵심: 연결 문자열로부터 클라이언트 객체만 생성합니다.
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # ⭐️ 제거된 부분: 아래 코드를 의도적으로 삭제하여 DB 서비스처럼 동작하게 만듭니다.
        # container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        # await container_client.get_container_properties() # <-- 이 부분이 미리 연결을 확인하는 범인이었습니다.
        
    except Exception as e:
        log.error(f"[STORAGE] 클라이언트 인스턴스 생성 실패: 연결 문자열 형식이 잘못되었을 수 있습니다. - {e}")
        blob_service_client = None


async def upload_image_and_get_url(file_bytes: bytes, original_filename: str) -> str:
    """
    이미지를 업로드하고 URL을 반환합니다. DB 서비스처럼 실제 작업 시점에 오류가 발생합니다.
    """
    global blob_service_client

    if blob_service_client is None:
        async with _init_lock:
            if blob_service_client is None:
                await _initialize_client()

    if not blob_service_client:
        raise ConnectionError("스토리지 클라이언트 인스턴스 생성에 실패했습니다. 연결 문자열을 확인하세요.")

    try:
        # 고유한 파일 이름 생성
        file_extension = os.path.splitext(original_filename)[1]
        blob_name = f"images/{uuid.uuid4()}{file_extension}"
        
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER_NAME,
            blob=blob_name
        )
        
        # ⭐️ 실제 업로드 시도: 만약 컨테이너 이름이나 네트워크 방화벽에 문제가 있다면
        # 바로 이 지점에서 오류가 발생할 것입니다. (예: ResourceNotFoundError)
        await blob_client.upload_blob(file_bytes, overwrite=True)
        
        log.info(f"[STORAGE] 이미지 업로드 성공. URL: {blob_client.url}")
        return blob_client.url

    except Exception as e:
        # DB 서비스와 마찬가지로, 실제 작업 중 발생하는 예외를 로깅하고 다시 발생시킵니다.
        log.error(f"[STORAGE] 오류: 이미지 업로드 중 실제 오류 발생 - {e.__class__.__name__}: {e}")
        raise e