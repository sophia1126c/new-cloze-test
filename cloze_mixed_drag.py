import random
import re
from typing import Dict, List, Tuple, Optional

import nltk
import streamlit as st
from docx import Document
from nltk import pos_tag, word_tokenize

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


# =========================
# 기본 설정
# =========================
MAX_BLANKS = 25
TOKEN_CANDIDATE_RE = re.compile(r"^[A-Za-z][A-Za-z\-']*$")


# =========================
# NLTK 데이터 준비
# =========================
def ensure_nltk_data() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass


ensure_nltk_data()


# =========================
# 유틸 함수
# =========================
def is_candidate_token(tok: str) -> bool:
    """
    빈칸 후보 단어 판별.
    평가용으로 너무 짧거나 문장부호/숫자 위주의 토큰은 제외.
    """
    token = tok.strip()

    if not TOKEN_CANDIDATE_RE.fullmatch(token):
        return False

    if len(token) <= 1:
        return False

    # 평가용으로 부적절한 너무 흔한 축약/기호성 토큰 방지
    excluded = {"'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}
    if token.lower() in excluded:
        return False

    return True


def tokenize_text(text: str) -> List[str]:
    return word_tokenize(text)


def assemble_tokens_plain(tokens: List[str]) -> str:
    """토큰을 일반 텍스트로 재조립"""
    out = ""
    no_space_before = {".", ",", "?", "!", ":", ";", "%", ")", "]", "}", "’", "'s", "n't"}
    no_space_after = {"(", "[", "{", "“", "‘", "$"}

    for i, token in enumerate(tokens):
        if i == 0:
            out += token
            continue

        prev = tokens[i - 1]

        if token in no_space_before or re.fullmatch(r"[^\w\s]", token):
            out += token
        elif prev in no_space_after:
            out += token
        else:
            out += " " + token

    return out


def read_docx_paragraphs(file_like) -> List[str]:
    doc = Document(file_like)
    return [para.text.strip() for para in doc.paragraphs]


def translate_paragraphs(paragraphs: List[str]) -> List[str]:
    if GoogleTranslator is None:
        return ["번역 기능을 사용할 수 없습니다. requirements.txt에 deep-translator가 있는지 확인하세요."]

    translator = GoogleTranslator(source="auto", target="ko")
    translated = []

    for para in paragraphs:
        if not para.strip():
            translated.append("")
            continue

        try:
            translated.append(translator.translate(para))
        except Exception:
            translated.append("번역 중 오류가 발생했습니다.")

    return translated


def make_seed(seed_input: str) -> int:
    """
    seed 입력값이 있으면 그 값을 사용하고,
    없으면 자동으로 6자리 seed 생성.
    """
    clean = seed_input.strip()

    if clean:
        try:
            return int(clean)
        except ValueError:
            return abs(hash(clean)) % 1_000_000

    return random.randint(100000, 999999)


def choose_non_adjacent_positions(
    all_candidates: List[Tuple[int, int]],
    target_count: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    같은 문단에서 바로 앞/뒤 단어가 동시에 빈칸이 되지 않게 선택.
    그래도 목표 개수에 부족하면 한 칸 간격 후보를 최대한 추가.
    """
    candidates = list(all_candidates)
    rng.shuffle(candidates)

    chosen: List[Tuple[int, int]] = []
    blocked = set()

    for para_idx, tok_idx in candidates:
        if (para_idx, tok_idx) in blocked:
            continue

        chosen.append((para_idx, tok_idx))

        blocked.add((para_idx, tok_idx - 1))
        blocked.add((para_idx, tok_idx))
        blocked.add((para_idx, tok_idx + 1))

        if len(chosen) >= target_count:
            break

    return chosen


def make_blank_html(num: int) -> str:
    return (
        f'<span style="white-space:nowrap;">'
        f'<b>({num})</b>'
        f'<span style="display:inline-block; min-width:88px; border-bottom:2px solid #222; margin:0 4px;">&nbsp;</span>'
        f'</span>'
    )


def assemble_tokens_html(tokens: List[str], blank_lookup: Dict[int, int]) -> str:
    """
    문제지 표시용 HTML 생성.
    blank_lookup: 토큰 인덱스 -> 빈칸 번호
    """
    out = ""
    no_space_before = {".", ",", "?", "!", ":", ";", "%", ")", "]", "}", "’", "'s", "n't"}
    no_space_after = {"(", "[", "{", "“", "‘", "$"}

    for i, token in enumerate(tokens):
        display = make_blank_html(blank_lookup[i]) if i in blank_lookup else token

        if i == 0:
            out += display
            continue

        prev = tokens[i - 1]

        if token in no_space_before or re.fullmatch(r"[^\w\s]", token):
            out += display
        elif prev in no_space_after:
            out += display
        else:
            out += " " + display

    return out


# =========================
# 문제 생성 함수
# =========================
def generate_questions_from_docx(
    file_like,
    seed: int,
    max_blanks: int = MAX_BLANKS,
) -> Tuple[List[str], Dict[int, str], List[str], List[str]]:
    """
    혼합형 cloze test 생성.
    - 품사 제한 없음
    - 최대 25개 빈칸
    - 연속 빈칸 방지
    - seed 기반 재현 가능
    """
    rng = random.Random(seed)

    original_paragraphs = read_docx_paragraphs(file_like)

    tokenized_paragraphs: List[List[str]] = []
    all_candidates: List[Tuple[int, int]] = []

    for para_idx, orig_text in enumerate(original_paragraphs):
        if not orig_text:
            tokenized_paragraphs.append([])
            continue

        tokens = tokenize_text(orig_text)
        tokenized_paragraphs.append(tokens)

        try:
            tagged = pos_tag(tokens)
        except Exception:
            tagged = [(t, "NN") for t in tokens]

        for tok_idx, (tok, tag) in enumerate(tagged):
            if is_candidate_token(tok):
                all_candidates.append((para_idx, tok_idx))

    n_candidates = len(all_candidates)

    if n_candidates == 0:
        return original_paragraphs, {}, original_paragraphs, []

    requested_blanks = min(max_blanks, n_candidates)
    chosen_positions = choose_non_adjacent_positions(all_candidates, requested_blanks, rng)
    chosen_positions_set = set(chosen_positions)

    answer_map: Dict[int, str] = {}
    question_paragraphs_html: List[str] = []
    next_blank_num = 1

    for para_idx, tokens in enumerate(tokenized_paragraphs):
        if not tokens:
            question_paragraphs_html.append("")
            continue

        blank_lookup: Dict[int, int] = {}

        for tok_idx, original_word in enumerate(tokens):
            if (para_idx, tok_idx) in chosen_positions_set:
                blank_lookup[tok_idx] = next_blank_num
                answer_map[next_blank_num] = original_word
                next_blank_num += 1

        question_paragraphs_html.append(assemble_tokens_html(tokens, blank_lookup))

    word_bank = list(answer_map.values())
    rng.shuffle(word_bank)

    return question_paragraphs_html, answer_map, original_paragraphs, word_bank


# =========================
# 채점 함수
# =========================
def grade_answers(answer_map: Dict[int, str]) -> Tuple[int, int, List[Dict[str, object]]]:
    total = len(answer_map)
    correct_count = 0
    results = []

    for num in sorted(answer_map.keys()):
        correct = answer_map[num]
        user_ans = st.session_state.get(f"answer_{num}", "")

        user_norm = str(user_ans).strip().lower()
        correct_norm = correct.strip().lower()

        is_correct = user_norm == correct_norm and user_norm != ""

        if is_correct:
            correct_count += 1

        results.append(
            {
                "num": num,
                "correct": correct,
                "user": user_ans,
                "is_correct": is_correct,
            }
        )

    return correct_count, total, results


def reset_quiz_state() -> None:
    keys_to_delete = [
        "questions",
        "answer_map",
        "original_paragraphs",
        "translated_paragraphs",
        "word_bank",
        "current_seed",
    ]

    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    for key in list(st.session_state.keys()):
        if key.startswith("answer_"):
            del st.session_state[key]


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Mixed Cloze Test", layout="wide")

st.title("📘 Mixed Cloze Test")

st.markdown(
    """
Word(.docx) 지문을 업로드하면 **혼합형 cloze test**를 생성합니다.  
빈칸에 들어갈 단어는 **단어은행**에서 선택할 수 있으며, 전체 빈칸 수는 **최대 25개**로 자동 생성됩니다.  
각 문제는 **고유 seed 번호**로 다시 재현할 수 있습니다.
"""
)

col_class, col_name = st.columns(2)

with col_class:
    class_name = st.text_input("반", value="", placeholder="예: 고2 3반")

with col_name:
    student_name = st.text_input("이름", value="", placeholder="예: 홍길동")

st.markdown("---")

st.header("⚙️ 문제 설정")
st.info("문항 유형은 전체 cloze test 중 **혼합형**으로 고정됩니다. 특정 품사만 따로 제한하지 않습니다.")
st.info("빈칸 수는 자동 생성되며 최대 25개로 고정됩니다. 연속된 단어가 동시에 빈칸이 되지 않도록 설정되어 있습니다.")

seed_input = st.text_input(
    "기존 문제 재현용 seed",
    value="",
    placeholder="이전에 생성된 seed 번호를 입력하면 같은 문제를 다시 만들 수 있습니다.",
)

show_translation = st.checkbox("문제 생성 후 한글 해석도 함께 제공", value=True)

uploaded_file = st.file_uploader("Word(.docx) 파일 업로드", type=["docx"])

col_reset, col_make = st.columns([1, 4])

with col_reset:
    if st.button("🧹 초기화"):
        reset_quiz_state()
        st.rerun()

with col_make:
    if st.button("📄 문제 만들기"):
        if uploaded_file is None:
            st.warning("먼저 Word(.docx) 파일을 업로드하세요.")
        else:
            try:
                uploaded_file.seek(0)

                seed = make_seed(seed_input)
                st.session_state["current_seed"] = seed

                questions, answer_map, original_paragraphs, word_bank = generate_questions_from_docx(
                    uploaded_file,
                    seed=seed,
                    max_blanks=MAX_BLANKS,
                )

                st.session_state["questions"] = questions
                st.session_state["answer_map"] = answer_map
                st.session_state["original_paragraphs"] = original_paragraphs
                st.session_state["word_bank"] = word_bank

                if show_translation:
                    st.session_state["translated_paragraphs"] = translate_paragraphs(original_paragraphs)
                else:
                    st.session_state["translated_paragraphs"] = []

                st.success(f"문제가 생성되었습니다. 빈칸 수: {len(answer_map)}개 / 최대 {MAX_BLANKS}개")
                st.code(str(seed), language="text")
                st.caption("위 seed 번호를 저장해 두면 같은 docx 파일로 동일한 문제를 다시 생성할 수 있습니다.")

            except Exception as e:
                st.error("문제 생성 중 오류가 발생했습니다.")
                st.exception(e)

st.markdown("---")


# =========================
# 사이드바 문제지
# =========================
with st.sidebar:
    st.header("📝 문제지")

    if "questions" in st.session_state:
        if "current_seed" in st.session_state:
            st.caption(f"Seed: {st.session_state['current_seed']}")

        for para in st.session_state["questions"]:
            if para.strip():
                st.markdown(para, unsafe_allow_html=True)
            else:
                st.write("")
    else:
        st.caption("문제지가 여기에 표시됩니다.")


# =========================
# 메인 문제 출력 및 답안 입력
# =========================
if "answer_map" not in st.session_state:
    st.info("문제지를 먼저 생성해 주세요.")
else:
    answer_map = st.session_state["answer_map"]
    word_bank = st.session_state.get("word_bank", [])

    if len(answer_map) == 0:
        st.warning("생성된 빈칸이 없습니다. 다른 지문을 사용해 보세요.")
    else:
        if "current_seed" in st.session_state:
            st.info(f"이 문제의 고유 seed 번호: **{st.session_state['current_seed']}**")

        st.header("📄 문제지")

        for para in st.session_state["questions"]:
            if para.strip():
                st.markdown(para, unsafe_allow_html=True)
            else:
                st.write("")

        st.markdown("---")

        st.subheader("🏦 단어은행")
        st.markdown(" / ".join([f"`{word}`" for word in word_bank]))

        st.markdown("---")

        st.subheader("✏️ 번호별 콤보박스 선택")
        st.caption("각 번호 옆의 콤보박스에서 정답 단어를 바로 선택하세요.")

        nums = sorted(answer_map.keys())
        for start in range(0, len(nums), 5):
            cols = st.columns(5)
            for col, num in zip(cols, nums[start:start + 5]):
                with col:
                    st.selectbox(
                        label=f"{num}번",
                        options=[""] + word_bank,
                        key=f"answer_{num}",
                        index=0,
                    )

        if st.button("✅ 채점하기"):
            correct_count, total, results = grade_answers(answer_map)
            score_pct = (correct_count / total) * 100 if total > 0 else 0.0

            st.markdown("---")
            st.subheader("📊 채점 결과")
            st.write(f"총 {total}문항 중 **{correct_count}개** 정답입니다.")
            st.write(f"점수: **{score_pct:.1f}점 / 100점**")

            for result in results:
                num = result["num"]
                correct = result["correct"]
                user_ans = result["user"]

                if result["is_correct"]:
                    st.success(f"{num}번: 정답! 입력: {user_ans}")
                else:
                    if str(user_ans).strip() == "":
                        st.error(f"{num}번: 무응답. 정답은 **{correct}** 입니다.")
                    else:
                        st.error(f"{num}번: 오답. 입력: `{user_ans}`, 정답: **{correct}**")

        translated_paragraphs = st.session_state.get("translated_paragraphs", [])

        if translated_paragraphs:
            st.markdown("---")
            with st.expander("📖 한글 해석 보기"):
                original_paragraphs = st.session_state.get("original_paragraphs", [])

                for idx, (eng, kor) in enumerate(zip(original_paragraphs, translated_paragraphs), start=1):
                    if not eng.strip():
                        continue

                    st.markdown(f"**[{idx}] 영문**")
                    st.write(eng)

                    st.markdown(f"**[{idx}] 한글 해석**")
                    st.write(kor)

                    st.markdown("---")
