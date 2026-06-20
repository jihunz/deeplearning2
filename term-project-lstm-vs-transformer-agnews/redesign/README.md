# 발표자료 디자인 가이드 & 빌드

LSTM vs Transformer 발표자료(`../deliverables/5_presentation_v2.pptx`)의 생성·수정 가이드.
디자인은 **tiny-budget-ada 프로젝트의 디자인 시스템**을 따른다 (불릿 본문 + 부분 강조 + 메타 프레임 표지).

## 파일 위치

| 파일 | 역할 |
|---|---|
| `redesign/build.py` | 슬라이드 빌드 (python-pptx). **본문 내용 수정은 여기** |
| `redesign/figures.py` | 그림 4개 생성 (matplotlib) |
| `redesign/fig_*.png` | 생성된 그림 (loss / confusion / ablation / mechanism) |
| `../deliverables/5_presentation_v2.pptx` | **최종 출력** |

전체 디자인 시스템 문서(원본, 별도 repo): `~/Downloads/논문/tiny-budget-ada/docs/ppt-design-system.md`

## 빌드 방법

```bash
PY=/Users/jihunjang/miniconda3/bin/python
$PY redesign/figures.py    # 그림 재생성 (그림 데이터·스타일 바꿀 때만)
$PY redesign/build.py      # 슬라이드 빌드 → ../deliverables/5_presentation_v2.pptx
```

의존: `python-pptx`, `Pillow`, `matplotlib` (miniconda 환경). 폰트 Pretendard: `~/Library/Fonts/Pretendard-*.ttf`.

## 디자인 시스템 (요약)

- **폰트**: Pretendard
- **색**: ink `#111827`(제목) / body `#374151`(본문) / sub `#6B7280`(보조) / **blue `#1D4ED8`(강조 1색)** / green `#0F766E`(성공)
- **표지** (메타 프레임): 좌측 세로 blue 액센트 + 상단 메타바(카테고리·연도) + 대형 제목(강조어 blue) + 부제 + 발표자/소속
- **본문**: 헤더(blue Bold) + **PowerPoint 기본 불릿 `•`**(색·크기 텍스트 상속) + 들여쓰기. **부분 강조**(단어·구 단위 bold·색)
- **진행 표시**: 배경 · 방법 · 결과 · 결론 (현재 단계만 blue Bold)
- **그림**: Transformer = blue / LSTM = 회색, 범례 대신 직접 라벨 + 인사이트 주석, 미니멀 축
- 강조는 절제(단일 blue), 디자인용 빈 줄·여백 금지, 콘텐츠 상단 정렬(valign top)

## 본문 수정 방법

`build.py`의 각 슬라이드 `body(...)` 호출에서 `blocks`만 편집한다.

```python
body(s, MX, 1.55, W, 5.2, [
  {'h':'헤더', 'items':[
     # 한 불릿 = run 튜플 리스트 → 한 불릿 안에서 부분 강조
     [("강조 텍스트", True, '111827'), (" 평문 부분",)],   # (텍스트, bold, 색hex)
     [("일반 항목",)],                                     # bold 생략=False, 색 생략=body
     {'lv':1, 'parts':[("들여쓴 서브 항목",)]},            # L1 들여쓰기(– 불릿)
  ]},
], 15, 14)   # 마지막 두 인자 = 본문 크기(fs), 헤더 그룹 간격(gap)
```

- **헤드라인**: `head(s, active, [(텍스트, 색hex), ...])` — `active` 0=배경 1=방법 2=결과 3=결론
- **그림**: `pic(s, path, x, y, box_w, box_h)` — 박스 안에서 aspect 유지(contain)
- **표지·도형·구분선**: `build.py` 상단 helper(`tbox`/`run`/`bullet`/`hline`/`rect`) 참조

수정 후 `python redesign/build.py` 재실행하면 pptx가 갱신된다.

## 주의

- 불릿은 **PowerPoint 기본 모양**(`•`, 텍스트색)이어야 함 — `bullet()`에서 `buClr`/`buSzPct`를 설정하지 말 것(색·크기 상속).
- 부분 강조는 **python-pptx에서만** 가능 (pptxgenjs는 한 문단에서 불릿+부분강조 동시 불가). pptxgenjs로 되돌리지 말 것.
- em dash(—) 사용 금지 → 콜론·쉼표·줄바꿈. 가운뎃점(·)은 한국어 병렬에만.
