
# 파일명: notification_service.py

import os
import time
import hmac
import hashlib
import base64
import httpx  # requests 대신 httpx를 임포트합니다.
import json
import requests
from urllib.parse import quote_plus, urlparse
from dotenv import load_dotenv

# --- 환경 변수 로드 및 설정 ---
load_dotenv()
CONNECTION_STRING = os.getenv("AZURE_NOTIFICATION_HUB_CONNECTION_STRING")
HUB_NAME = os.getenv("AZURE_NOTIFICATION_HUB_NAME")

if not CONNECTION_STRING or not HUB_NAME:
    raise ValueError("환경 변수 AZURE_NOTIFICATION_HUB_CONNECTION_STRING 또는 AZURE_NOTIFICATION_HUB_NAME이 설정되지 않았습니다.")

try:
    parsed_conn_str = {key.lower(): value for key, value in (part.split('=', 1) for part in CONNECTION_STRING.split(';'))}
    NAMESPACE = urlparse(parsed_conn_str['endpoint']).hostname.split('.')[0]
    SAS_KEY_NAME = parsed_conn_str['sharedaccesskeyname']
    SAS_KEY_VALUE = parsed_conn_str['sharedaccesskey']
    API_VERSION = "2020-06"
except (KeyError, IndexError):
    raise ValueError("연결 문자열(Connection String)의 형식이 올바르지 않습니다.")

# --- SAS 토큰 생성 함수 (이 함수는 동기로 유지해도 괜찮습니다) ---
def generate_sas_token(uri, key_name, key_value):
    expiry = int(time.time()) + 3600
    string_to_sign = quote_plus(uri, safe='') + '\n' + str(expiry)
    signature = hmac.new(key=key_value.encode('utf-8'), msg=string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
    encoded_signature = quote_plus(base64.b64encode(signature))
    token = f"SharedAccessSignature sr={quote_plus(uri, safe='')}&sig={encoded_signature}&se={str(expiry)}&skn={key_name}"
    return token

# --- [핵심] Installation 생성 또는 업데이트 함수 (비동기로 수정) ---
async def create_or_update_installation(installation_id, platform, device_token, tags):
    """지정된 정보로 Installation을 생성하거나 덮어씁니다 (비동기 버전)."""
    
    resource_uri = f"https://{NAMESPACE}.servicebus.windows.net/{HUB_NAME}/installations/{installation_id}"
    url = f"{resource_uri}?api-version={API_VERSION}"
    sas_token = generate_sas_token(resource_uri, SAS_KEY_NAME, SAS_KEY_VALUE)
    headers = {'Authorization': sas_token, 'Content-Type': 'application/json'}
    payload = {"installationId": installation_id, "platform": platform.lower(), "pushChannel": device_token, "tags": tags}
    
    print(f"Installation 등록/업데이트 요청 시작 (ID: {installation_id})")
    
    # 비동기 HTTP 클라이언트를 사용합니다.
    async with httpx.AsyncClient() as client:
        try:
            # client.put 앞에 'await'를 붙여 비동기적으로 요청을 보냅니다.
            response = await client.put(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            if response.status_code == 201:
                print(f"[성공] 새로운 Installation 생성 완료. Status Code: 201")
            elif response.status_code == 200:
                print(f"[성공] 기존 Installation 업데이트 완료. Status Code: 200")
            
            return True, response.status_code

        except httpx.HTTPStatusError as e:
            # httpx의 예외 처리를 사용합니다.
            status_code = e.response.status_code
            error_text = e.response.text
            print(f"[실패] 오류 발생! Status Code: {status_code}")
            print(f"오류 내용: {error_text}")
            return False, status_code
        



def send_notification_as_single_message(message: str, conversation_id: str):
    """
    등록된 모든 Apple 기기에게 제목 없는 단일 메시지 알림을 보냅니다.

    :param message: 알림으로 보낼 전체 문장.
    """
    resource_uri = f"https://{NAMESPACE}.servicebus.windows.net/{HUB_NAME}/messages"
    url = f"{resource_uri}?api-version={API_VERSION}"
    sas_token = generate_sas_token(resource_uri, SAS_KEY_NAME, SAS_KEY_VALUE)

    # 3. [수정] "alert" 딕셔너리에 "body"만 남깁니다.
    apple_payload = {
        "aps": {
            "alert": {
                "body": message  # title을 제거하고 body에 모든 내용을 담습니다.
            },
            "sound": "default",
            "badge": 1,
            "category": "FEELINK_CHAT",
        },
            "conversation_id": conversation_id

    }

    # 4. HTTP 헤더 설정 (태그 없음)
    headers = {
        'Authorization': sas_token,
        'Content-Type': 'application/json;charset=utf-8',
        'ServiceBusNotification-Format': 'apple'
    }

    print(f"단일 메시지 알림 보내기 시작 -> 내용: '{message}'")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(apple_payload))
        response.raise_for_status()

        print(f"[성공] 알림이 성공적으로 전송되었습니다.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[실패] 알림 전송 중 오류 발생!")
        print(f"오류 내용: {e.response.text if e.response else e}")
        return False