# 파일명: send_single_message.py

import os
import time
import hmac
import hashlib
import base64
import requests
import json
from urllib.parse import quote_plus, urlparse
from dotenv import load_dotenv

# --- 환경 변수 로드 및 설정 (기존과 동일) ---
load_dotenv()
CONNECTION_STRING = os.getenv("AZURE_NOTIFICATION_HUB_CONNECTION_STRING")
HUB_NAME = os.getenv("AZURE_NOTIFICATION_HUB_NAME")

if not CONNECTION_STRING or not HUB_NAME:
    raise ValueError("환경 변수가 올바르게 설정되지 않았습니다.")

try:
    parsed_conn_str = {key.lower(): value for key, value in (part.split('=', 1) for part in CONNECTION_STRING.split(';'))}
    NAMESPACE = urlparse(parsed_conn_str['endpoint']).hostname.split('.')[0]
    SAS_KEY_NAME = parsed_conn_str['sharedaccesskeyname']
    SAS_KEY_VALUE = parsed_conn_str['sharedaccesskey']
    API_VERSION = "2020-06"
except (KeyError, IndexError):
    raise ValueError("연결 문자열(Connection String)의 형식이 올바르지 않았습니다.")

# --- SAS 토큰 생성 함수 (기존과 동일) ---
def generate_sas_token(uri, key_name, key_value):
    expiry = int(time.time()) + 3600
    string_to_sign = quote_plus(uri, safe='') + '\n' + str(expiry)
    signature = hmac.new(key=key_value.encode('utf-8'), msg=string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
    encoded_signature = quote_plus(base64.b64encode(signature))
    return f"SharedAccessSignature sr={quote_plus(uri, safe='')}&sig={encoded_signature}&se={str(expiry)}&skn={key_name}"

# --- [핵심] 하나의 문장으로 알림을 보내는 함수 ---
def send_notification_as_single_message(message: str):
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
            "badge": 1
        }
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

# --- 스크립트 실행 부분 ---
if __name__ == "__main__":
    # ⭐️ 여기에 보낼 전체 메시지 내용을 수정하세요! ⭐️
    
    full_message = "안녕하세요. 감사합니다. 반가워요. 하지메마시때. 잘 부탁드립니다. 섹스"

    print("--- 보낼 알림 정보 ---")
    print(f"전체 메시지: {full_message}")
    print("--------------------")

    answer = input("정말로 이 알림을 모든 Apple 기기에 보내시겠습니까? (yes/no): ")
    if answer.lower() == 'yes':
        send_notification_as_single_message(full_message)
    else:
        print("알림 보내기 작업이 취소되었습니다.")