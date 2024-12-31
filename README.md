# AI_RAG_LLM
Making_AI_RAG_LLM

## 1. Layer
- 크게 4개의 layer로 만든 예정입니다.
  - controller
  - service
  - repository
  - entity

### controller
- `API`와 route를 지정할 수 있습니다

### Service
```
많은 로직들이 들어가는 layer이기 때문에 아래의 구조로 설계를 할까 합니다.
```
- Default Service Layer
  - Command service : 저장, 수정, 삭제
  - Query service : 조회
- Pipeline Service Layer
  - `Default Service Layer`를 조합하는 service layer입니다.

### Repository
현재 데이터베이스의 종류가 3가지 입니다. 따라서 각 DB에 호환되는 상위의 abstract repository를 정의합니다.
- PineconeDB
- MySQL
- MongoDB

#### Abstract Repository
각 DB의 특성에 맞게 정의를 합니다.

#### Domain Repository
- 각 DB Abstract Repository를 상속받아 정의합니다.
- 기본적인 Get, Update, Insert, Delete, FindALL, isExist 함수를 재정의합니다.


### Entity
- 각 DB table 스펙에 맞는 class를 정의합니다.
- 각 객체에 대한 책임을 가지며, table에 들어가는 데이터의 스펙과 validation check를 수행할 수 있도록 합니다.



