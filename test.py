import os
import time
import hmac
import hashlib
import base64
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse
from dotenv import load_dotenv

# --- 환경 변수 로드 및 설정 ---
load_dotenv()
CONNECTION_STRING = os.getenv("AZURE_NOTIFICATION_HUB_CONNECTION_STRING")
HUB_NAME = os.getenv("AZURE_NOTIFICATION_HUB_NAME")

if not CONNECTION_STRING or not HUB_NAME:
    raise ValueError("AZURE_NOTIFICATION_HUB_CONNECTION_STRING 또는 AZURE_NOTIFICATION_HUB_NAME 환경 변수를 확인해주세요.")

parsed_conn_str = {key.lower(): value for key, value in (part.split('=', 1) for part in CONNECTION_STRING.split(';'))}
NAMESPACE = urlparse(parsed_conn_str['endpoint']).hostname.split('.')[0]
SAS_KEY_NAME = parsed_conn_str['sharedaccesskeyname']
SAS_KEY_VALUE = parsed_conn_str['sharedaccesskey']
API_VERSION = "2020-06"

# --- SAS 토큰 생성 함수 ---
def generate_sas_token(uri, key_name, key_value):
    expiry = int(time.time()) + 3600
    string_to_sign = quote_plus(uri, safe='') + '\n' + str(expiry)
    signature = hmac.new(key=key_value.encode('utf-8'), msg=string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
    encoded_signature = quote_plus(base64.b64encode(signature))
    return f"SharedAccessSignature sr={quote_plus(uri, safe='')}&sig={encoded_signature}&se={str(expiry)}&skn={key_name}"

# --- 단일 Registration 삭제 함수 ---
def delete_registration(registration_id):
    resource_uri = f"https://{NAMESPACE}.servicebus.windows.net/{HUB_NAME}/registrations/{registration_id}"
    url = f"{resource_uri}?api-version={API_VERSION}"
    sas_token = generate_sas_token(resource_uri, SAS_KEY_NAME, SAS_KEY_VALUE)
    headers = {'Authorization': sas_token, 'If-Match': '*'}
    
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        print(f"[성공] Registration ID: {registration_id} 삭제 완료.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[실패] Registration ID: {registration_id} 삭제 중 오류: {e.response.status_code if e.response else 'N/A'}")
        return False

# --- 모든 Registrations 조회 및 삭제 실행 함수 ---
def clear_all_registrations():
    print("모든 Registrations 삭제 작업을 시작합니다...")
    
    # 1. 모든 Registrations 목록 조회
    get_resource_uri = f"https://{NAMESPACE}.servicebus.windows.net/{HUB_NAME}/registrations"
    get_url = f"{get_resource_uri}?api-version={API_VERSION}"
    get_sas_token = generate_sas_token(get_resource_uri, SAS_KEY_NAME, SAS_KEY_VALUE)
    get_headers = {'Authorization': get_sas_token, 'x-ms-version': API_VERSION}
    
    try:
        response = requests.get(get_url, headers=get_headers)
        response.raise_for_status()
        
        xml_root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'content': 'http://schemas.microsoft.com/netservices/2010/10/servicebus/connect'}
        
        registrations_to_delete = [desc.text for desc in xml_root.findall('.//content:RegistrationId', ns) if desc.text]

        if not registrations_to_delete:
            print("삭제할 Registration이 없습니다.")
            return

        print(f"총 {len(registrations_to_delete)}개의 Registration을 삭제 대상으로 찾았습니다.")
        
        # 2. 조회된 목록 기반으로 삭제
        for reg_id in registrations_to_delete:
            delete_registration(reg_id)
            time.sleep(0.1) # API Throttling 방지

        print("\n모든 Registrations 삭제 작업이 완료되었습니다.")

    except requests.exceptions.RequestException as e:
        print(f"Registration 목록 조회 중 오류 발생: {e.response.text if e.response else e}")

# --- 실행 ---
if __name__ == "__main__":
    answer = input("정말로 모든 Registrations를 삭제하시겠습니까? (yes/no): ")
    if answer.lower() == 'yes':
        clear_all_registrations()
    else:
        print("삭제 작업이 취소되었습니다.")