# feelink_back

Feelink는 시각장애인의 모바일 앱 접근성을 높이기 위한 화면 음성 안내
서비스입니다. 사용자는 스마트폰 후면을 두 번 탭하는 간단한 제스처로 현재 화면을
AI에게 전달하고, 화면 속 이미지와 텍스트, 주요 UI 요소에 대한 설명을 음성으로
안내받을 수 있습니다.

이 저장소는 Feelink 서비스의 백엔드입니다. iOS, Android 앱에서 전달한 화면 이미지와
사용자 질문을 받아 Gemini API에 맞게 프롬프팅하고, 시각장애인에게 필요한 방식으로
화면 설명과 후속 질의응답을 생성합니다. 업로드된 이미지는 Azure Blob Storage에
저장하고, 대화 기록은 DB에 저장하며, Azure Notification Hub를 통해 iOS 푸시 알림도
전송할 수 있습니다.

이 백엔드는 단순히 AI 응답을 중계하는 서버가 아니라, Feelink 서비스에서 필요한
데이터 저장과 클라우드 연동 구조를 구현해보기 위한 역할도 큽니다. 특히 대화 기록,
이미지 저장, 알림 전송처럼 앱 단독으로 처리하기 어려운 부분을 서버와 DB, Azure
서비스로 분리해 관리하는 것을 목표로 했습니다.

## 서비스 개요

> 두 번의 가벼운 탭, 세상을 소리로 다시 만나다.

시각장애인은 스마트폰을 사용할 때 작은 글자, 복잡한 메뉴 구조, 이미지 중심의 정보
때문에 많은 제약을 겪습니다. 기존 모바일 환경에서는 이미지 대체 텍스트가 없거나,
화면의 시각 정보가 음성 안내로 충분히 전달되지 않아 앱을 사용하는 데 어려움이
있습니다.

Feelink는 후면 더블 탭으로 현재 화면을 인식하고, AI가 화면 정보를 분석한 뒤 음성으로
안내하는 보조 서비스입니다. 사용자는 단순한 제스처만으로 화면 내용을 들을 수 있고,
궁금한 점은 음성 또는 챗봇 형태로 이어서 질문할 수 있습니다.

## 핵심 아이디어

```text
후면 더블 탭 -> 화면 인식 -> AI 분석 -> 음성 안내
```

주제는 "간단한 제스처로 화면을 음성으로 들려주는 보조 서비스"입니다. 기존 스크린
리더가 읽어주지 못하는 이미지, 복잡한 UI, 맥락이 필요한 화면 정보를 AI가 보완해서
설명하는 것이 핵심입니다.

## 목표

- 이미지 대체 텍스트 부재로 인한 시각장애인의 정보 불균형을 해소합니다.
- 화면 속 이미지, 텍스트, UI 정보를 효과적으로 설명해 정보 접근 장벽을 낮춥니다.
- 두 번의 탭과 음성 기반 AI 챗봇 인터랙션으로 누구나 쉽게 사용할 수 있는 접근성을 제공합니다.
- 필요한 정보를 실시간에 가깝게 안내하고, 궁금한 점을 바로 질문하고 답변받을 수 있도록 지원합니다.

## 주요 기능

- 실시간 화면 분석 및 설명
- 이미지 및 텍스트 정보의 자동 음성 안내
- 사용자 질문에 대한 대화형 AI 응답
- 사용자 맞춤형 음성 설정 지원
- iOS 푸시 알림을 통한 AI 답변 전달

## 앱 구조

Feelink는 iOS와 Android 앱을 함께 고려한 서비스입니다.

- iOS 앱은 Swift로 개발합니다.
- Android 앱은 Kotlin으로 개발합니다.
- Android에서는 투명 오버레이 탭을 띄워 사용자가 현재 앱 흐름을 크게 방해받지 않고
  화면 분석을 실행할 수 있도록 구성합니다.
- iOS에서는 Azure Notification Hub를 활용해 AI 답변을 알림으로 보내고, 사용자가
  시스템 음성 안내를 통해 답변을 들을 수 있도록 합니다.

중요한 설계 방향은 기존 모바일 접근성 기능과 함께 사용할 수 있도록 하는 것입니다.
iOS의 VoiceOver, Android의 TalkBack 같은 스크린 리더를 사용하다가 이미지나 복잡한
화면을 만나면, 후면 더블 탭으로 Feelink를 호출해 기존 음성 안내를 끊고 현재 화면에
대한 AI 설명을 들을 수 있도록 하는 구조를 목표로 합니다.

## 백엔드 역할

- 모바일 앱에서 업로드한 화면 이미지를 수신합니다.
- 화면 해설에 특화된 시스템 프롬프트를 적용해 Gemini 모델에 질의합니다.
- 첫 화면 설명과 후속 질문을 구분해 대화 맥락을 유지합니다.
- 사용자의 질문과 AI 답변을 DB에 저장해 이후 대화에서 다시 활용합니다.
- 이미지 파일을 Azure Blob Storage에 저장합니다.
- 테스트 흐름에서는 AI 답변을 Azure Notification Hub를 통해 푸시 알림으로 전송합니다.
- Azure 기반 배포, 저장소, DB, 알림 연동 구조를 백엔드에서 실험하고 구현합니다.

## 기술 스택

- Python
- FastAPI
- Google Gemini API
- Microsoft Azure App Service
- Azure SQL 또는 PostgreSQL 기반 DB
- SQLAlchemy async
- Azure Blob Storage
- Azure Notification Hub

## 프로젝트 구조

```text
.
├── main.py                    # FastAPI 앱 진입점
├── router.py                  # HTTP 라우터 정의
├── requirements.txt           # Python 의존성 목록
└── api
    ├── ai_service.py          # Gemini 채팅 및 이미지 분석 로직
    ├── db_service.py          # DB 대화 CRUD
    ├── models.py              # SQLAlchemy 모델
    ├── notification_service.py# Azure Notification Hub 연동
    └── storage_service.py     # Azure Blob Storage 업로드 로직
```

## 환경 변수

프로젝트 루트에 `.env` 파일을 생성합니다. 현재 코드의 DB 연결은 SQLAlchemy async와
`asyncpg` 드라이버를 기준으로 작성되어 있어, 아래 예시는 PostgreSQL 형식입니다.
Azure SQL로 구성하는 경우에는 DB 드라이버와 연결 문자열을 Azure SQL 환경에 맞게
조정해야 합니다.

```env
GOOGLE_API_KEY=your_google_gemini_api_key
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/database

AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
AZURE_STORAGE_CONTAINER_NAME=your_blob_container_name

AZURE_NOTIFICATION_HUB_CONNECTION_STRING=your_notification_hub_connection_string
AZURE_NOTIFICATION_HUB_NAME=your_notification_hub_name
```

앱 시작 시 AI, DB, 알림 모듈을 import하므로 필수 환경 변수가 없으면 서버가 실행되지
않을 수 있습니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
uvicorn main:app --reload
```

기본 로컬 주소:

```text
http://127.0.0.1:8000
```

헬스 체크:

```bash
curl http://127.0.0.1:8000/
```

예상 응답:

```json
{"status":"ok"}
```

## API

### `POST /start_chat`

새 이미지 채팅 세션을 시작합니다.

폼 필드:

- `user_question`: 사용자 질문
- `image_file`: 업로드할 이미지 파일

예시:

```bash
curl -X POST http://127.0.0.1:8000/start_chat \
  -F "user_question=이 화면을 설명해줘" \
  -F "image_file=@/path/to/image.png"
```

응답:

```json
{
  "conversation_id": "uuid",
  "answer": "AI answer text"
}
```

### `POST /continue_chat`

기존 채팅 세션을 이어서 진행합니다.

폼 필드:

- `conversation_id`: `/start_chat`에서 받은 대화 UUID
- `user_question`: 후속 질문

예시:

```bash
curl -X POST http://127.0.0.1:8000/continue_chat \
  -F "conversation_id=00000000-0000-0000-0000-000000000000" \
  -F "user_question=버튼은 어디에 있어?"
```

응답:

```json
{
  "conversation_id": "uuid",
  "answer": "AI answer text"
}
```

### `POST /register_device`

Azure Notification Hub에 디바이스 installation을 등록하거나 업데이트합니다.

폼 필드:

- `device_token`: APNs 또는 FCM 디바이스 토큰
- `installation_id`: 디바이스 installation ID, 기본값은 `123`
- `platform`: 푸시 플랫폼, 기본값은 `apns`
- `tags`: 선택값, 쉼표로 구분한 태그 목록

예시:

```bash
curl -X POST http://127.0.0.1:8000/register_device \
  -F "installation_id=device-uuid" \
  -F "platform=apns" \
  -F "device_token=apns-device-token" \
  -F "tags=test,ios"
```

### 테스트 알림 엔드포인트

아래 엔드포인트는 채팅 흐름과 동일하게 동작하면서, AI 답변을 푸시 알림으로 보내는
백그라운드 작업도 함께 등록합니다.

- `POST /test`
- `POST /continue_test`

`/test`와 `/continue_test`는 응답으로 `conversation_id`만 반환합니다.

## 데이터베이스 모델

SQLAlchemy 모델은 아래 테이블을 정의합니다.

- `users`
- `fcm_devices`
- `conversations`

대화 기록은 `conversations.history` JSONB 컬럼에 저장됩니다.

## 참고

- 업로드된 이미지는 Azure Blob Storage의 `images/` prefix 아래에 저장됩니다.
- 채팅을 이어갈 때는 저장된 Blob URL에서 원본 이미지를 다시 불러옵니다.
- DB 저장 및 업데이트는 AI 응답 생성 이후 FastAPI background task로 예약됩니다.
