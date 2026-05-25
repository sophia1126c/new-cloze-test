import random
import re
from typing import Dict, List, Tuple

import nltk
import streamlit as st
from docx import Document
from nltk import pos_tag, word_tokenize

try:
    from streamlit_sortables import sort_items
except Exception:
    sort_items = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


# ---------- NLTK data ----------
def ensure_nltk_data() -> None:
    """Download required NLTK data only when missing."""
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

# ---------- constants ----------
TOKEN_CANDIDATE_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")
PUNCT_RE = re.compile(r"[^\w\s]")


# ---------- utility functions ----------
def is_candidate_token(tok: str) -> bool:
    """Return True if a token can reasonably become a blank."""
    return bool(TOKEN_CANDIDATE_RE.fullmatch(tok))


def tokenize_preserve_spacing(text: str) -> List[str]:
    return word_tokenize(text)


def assemble_tokens(tokens: List[str]) -> str:
    """Reassemble tokens into a readable paragraph."""
    out = ""
    no_space_before = {".", ",", "?", "!", ":", ";", ")", "]", "}", "'", "\""}
    no_space_after = {"(", "[", "{", "'", "\""}

    for i, token in enumerate(tokens):
        if i == 0:
            out += token
            continue

        prev = tokens[i - 1]
        if token in no_space_before or PUNCT_RE.fullmatch(token):
            out += token
        elif prev in no_space_after:
            out += token
        else:
            out += " " + token
    return out


def make_display_blank(num: int, original_word: str) -> str:
    underline = "＿" * max(4, len(original_word))
    return f"({num}){underline}"


def normalize_answer(text: str) -> str:
    return text.strip().lower()


# ---------- translation ----------
def translate_paragraphs_to_korean(paragraphs: List[str]) -> List[str]:
    """Try automatic Korean translation. Falls back to an explanatory message."""
    if GoogleTranslator is None:
        return [
            "자동 번역 라이브러리(deep-translator)가 설치되어 있지 않아 번역을 생성하지 못했습니다."
            for _ in paragraphs
        ]

    translated: List[str] = []
    translator = GoogleTranslator(source="auto", target="ko")

    for para in paragraphs:
        if not para.strip():
            translated.append("")
            continue
        try:
            translated.append(translator.translate(para))
        except Exception:
            translated.append("자동 번역에 실패했습니다. 인터넷 연결 또는 번역 서버 상태를 확인하세요.")

    return translated


# ---------- question generation ----------
def generate_mixed_cloze_from_docx(
    file_like,
    blank_ratio_fraction: float,
    seed: int | None = None,
) -> Tuple[List[str], List[str], Dict[int, str], List[str]]:
    """
    Generate a mixed-type cloze test.

    Returns:
        question_paragraphs: paragraphs with numbered blanks
        original_paragraphs: original English paragraphs
        answer_map: {blank_number: correct answer}
        word_bank: shuffled list of answer words
    """
    if seed is not None:
        random.seed(seed)

    src = Document(file_like)
    question_paragraphs: List[str] = []
    original_paragraphs: List[str] = []
    answer_map: Dict[int, str] = {}
    next_blank_num = 1

    for para in src.paragraphs:
        orig_text = para.text.strip()
        if not orig_text:
            question_paragraphs.append("")
            original_paragraphs.append("")
            continue

        original_paragraphs.append(orig_text)
        tokens = tokenize_preserve_spacing(orig_text)

        try:
            tagged = pos_tag(tokens)
        except Exception:
            tagged = [(token, "NN") for token in tokens]

        # Mixed type: do not limit by POS. Any meaningful word can become a blank.
        candidate_indices = [
            i for i, (token, _tag) in enumerate(tagged) if is_candidate_token(token)
        ]

        n_candidates = len(candidate_indices)
        n_blanks = max(1, int(round(n_candidates * blank_ratio_fraction))) if n_candidates else 0
        n_blanks = min(n_blanks, n_candidates)

        chosen_indices = random.sample(candidate_indices, n_blanks) if n_blanks else []
        out_tokens = list(tokens)

        for idx in sorted(chosen_indices):
            original_word = tokens[idx]
            out_tokens[idx] = make_display_blank(next_blank_num, original_word)
            answer_map[next_blank_num] = original_word
            next_blank_num += 1

        question_paragraphs.append(assemble_tokens(out_tokens))

    word_bank = list(answer_map.values())
    random.shuffle(word_bank)
    return question_paragraphs, original_paragraphs, answer_map, word_bank


# ---------- grading ----------
def get_user_answers_from_drag_order(answer_map: Dict[int, str], ordered_words: List[str]) -> Dict[int, str]:
    answers: Dict[int, str] = {}
    for idx, num in enumerate(sorted(answer_map.keys())):
        answers[num] = ordered_words[idx] if idx < len(ordered_words) else ""
    return answers


def get_user_answers_from_selectboxes(answer_map: Dict[int, str]) -> Dict[int, str]:
    answers: Dict[int, str] = {}
    for num in sorted(answer_map.keys()):
        answers[num] = st.session_state.get(f"answer_{num}", "")
    return answers


def grade_answers(answer_map: Dict[int, str], user_answers: Dict[int, str]):
    total = len(answer_map)
    correct_count = 0
    results = []

    for num in sorted(answer_map.keys()):
        correct = answer_map[num]
        user_ans = user_answers.get(num, "")
        is_correct = normalize_answer(user_ans) == normalize_answer(correct) and user_ans.strip() != ""
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


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Mixed Cloze Test", layout="wide")

st.title("📘 Mixed Cloze Test")
st.markdown(
    "Word(.docx) 지문을 업로드하면 **혼합형 cloze test**를 생성합니다. "
    "빈칸에 들어갈 단어는 단어은행에서 드래그하여 순서대로 배열하거나, 선택형으로 입력할 수 있습니다."
)

col_class, col_name = st.columns(2)
with col_class:
    class_name = st.text_input("반", value="", placeholder="예: 고2 3반")
with col_name:
    student_name = st.text_input("이름", value="", placeholder="예: 홍길동")

st.markdown("---")

st.subheader("⚙️ 문제 설정")
st.info("문항 유형은 전체 cloze test 중 **혼합형**으로 고정됩니다. 특정 품사만 따로 제한하지 않습니다.")

col_ratio, col_seed = st.columns(2)
with col_ratio:
    blank_pct = st.slider("빈칸 비율 (%)", min_value=5, max_value=80, value=20, step=5)
with col_seed:
    seed_text = st.text_input("랜덤 seed 선택 사항", value="", placeholder="같은 문제를 다시 만들고 싶을 때 숫자 입력")

answer_mode = st.radio(
    "답안 입력 방식",
    ["드래그해서 순서 배열", "빈칸별 선택"],
    horizontal=True,
)

show_translation_option = st.checkbox("문제 생성 후 한글 해석도 함께 제공", value=True)

uploaded_file = st.file_uploader("Word(.docx) 파일 업로드", type=["docx"])

if st.button("🧹 초기화"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if uploaded_file is not None:
    if st.button("📄 문제 만들기"):
        try:
            uploaded_file.seek(0)
            seed = int(seed_text) if seed_text.strip().isdigit() else None
            questions, originals, answer_map, word_bank = generate_mixed_cloze_from_docx(
                uploaded_file,
                blank_pct / 100.0,
                seed=seed,
            )

            translations = translate_paragraphs_to_korean(originals) if show_translation_option else []

            st.session_state["questions"] = questions
            st.session_state["originals"] = originals
            st.session_state["translations"] = translations
            st.session_state["answer_map"] = answer_map
            st.session_state["word_bank"] = word_bank
            st.session_state["answer_mode"] = answer_mode
            st.success("문제가 생성되었습니다. 왼쪽 문제지를 보면서 단어은행의 단어를 배열하세요.")
        except Exception as e:
            st.error("문제 생성 중 오류가 발생했습니다.")
            st.exception(e)
else:
    st.info("먼저 Word(.docx) 파일을 업로드하세요.")

st.markdown("---")

# ---------- Sidebar: test paper ----------
with st.sidebar:
    st.header("📝 문제지")
    if "questions" in st.session_state:
        for para in st.session_state["questions"]:
            if para.strip():
                st.markdown(para)
            else:
                st.write("")
    else:
        st.caption("docx 파일을 업로드한 뒤 '문제 만들기'를 누르면 여기에 문제지가 표시됩니다.")

# ---------- Main: answer area ----------
if "answer_map" in st.session_state:
    answer_map = st.session_state["answer_map"]
    word_bank = st.session_state["word_bank"]

    if not answer_map:
        st.warning("생성된 빈칸이 없습니다. 빈칸 비율을 올리거나 다른 지문을 사용해 보세요.")
    else:
        st.subheader("🧩 단어은행")
        st.caption("드래그 방식에서는 단어를 1번 빈칸부터 마지막 빈칸까지 들어갈 순서대로 배열하세요.")

        if st.session_state.get("answer_mode", answer_mode) == "드래그해서 순서 배열" and sort_items is not None:
            ordered_words = sort_items(
                word_bank,
                direction="horizontal",
                key="drag_word_bank",
            )
            st.session_state["ordered_words"] = ordered_words

            st.markdown("#### 현재 배열")
            preview_cols = st.columns(min(4, max(1, len(answer_map))))
            for i, num in enumerate(sorted(answer_map.keys())):
                with preview_cols[i % len(preview_cols)]:
                    selected_word = ordered_words[i] if i < len(ordered_words) else ""
                    st.write(f"**{num}번** → {selected_word}")

            user_answers = get_user_answers_from_drag_order(answer_map, ordered_words)

        else:
            if st.session_state.get("answer_mode", answer_mode) == "드래그해서 순서 배열" and sort_items is None:
                st.warning(
                    "streamlit-sortables가 설치되어 있지 않아 선택형 입력으로 전환합니다. "
                    "requirements.txt에 streamlit-sortables를 추가하세요."
                )

            st.markdown("#### 빈칸별 선택")
            options = [""] + word_bank
            for num in sorted(answer_map.keys()):
                st.selectbox(
                    label=f"{num}번",
                    options=options,
                    key=f"answer_{num}",
                )
            user_answers = get_user_answers_from_selectboxes(answer_map)

        if st.button("✅ 채점하기"):
            correct_count, total, results = grade_answers(answer_map, user_answers)
            score_pct = (correct_count / total) * 100 if total else 0.0

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
                    if user_ans.strip() == "":
                        st.error(f"{num}번: 무응답. 정답은 **{correct}** 입니다.")
                    else:
                        st.error(f"{num}번: 오답. 입력: `{user_ans}`, 정답: **{correct}**")

        if st.session_state.get("translations"):
            st.markdown("---")
            with st.expander("📖 지문 한글 해석 보기", expanded=False):
                originals = st.session_state.get("originals", [])
                translations = st.session_state.get("translations", [])

                for i, original in enumerate(originals, start=1):
                    if not original.strip():
                        continue
                    korean = translations[i - 1] if i - 1 < len(translations) else ""
                    st.markdown(f"**문단 {i} 원문**")
                    st.write(original)
                    st.markdown(f"**문단 {i} 해석**")
                    st.write(korean)
else:
    st.info("문제지를 먼저 생성해 주세요.")
