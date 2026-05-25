import random
import re
from typing import Dict, List, Tuple

import nltk
import streamlit as st
from docx import Document
from nltk import pos_tag, word_tokenize

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


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
# 기본 설정
# =========================
MAX_BLANKS = 25
TOKEN_CANDIDATE_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")


def is_candidate_token(tok: str) -> bool:
    return bool(TOKEN_CANDIDATE_RE.search(tok))


def tokenize_preserve_spacing(text: str) -> List[str]:
    return word_tokenize(text)


def assemble_tokens(tokens: List[str]) -> str:
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


def generate_questions_from_docx(
    file_like,
    random_seed: str = "",
    max_blanks: int = MAX_BLANKS,
) -> Tuple[List[str], Dict[int, str], List[str], List[str]]:
    """
    혼합형 cloze test 생성.
    품사 제한 없이 전체 단어를 후보로 삼되, 전체 빈칸 수는 max_blanks 이하로 제한.
    """
    original_paragraphs = read_docx_paragraphs(file_like)

    if random_seed.strip():
        random.seed(random_seed.strip())

    tokenized_paragraphs = []
    all_candidates = []

    for para_idx, orig_text in enumerate(original_paragraphs):
        if not orig_text:
            tokenized_paragraphs.append([])
            continue

        tokens = tokenize_preserve_spacing(orig_text)
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

    requested_blanks = min(25, n_candidates)

    chosen_positions = set(random.sample(all_candidates, requested_blanks))

    answer_map: Dict[int, str] = {}
    question_paragraphs: List[str] = []
    next_blank_num = 1

    for para_idx, tokens in enumerate(tokenized_paragraphs):
        if not tokens:
            question_paragraphs.append("")
            continue

        out_tokens = list(tokens)

        for tok_idx, original_word in enumerate(tokens):
            if (para_idx, tok_idx) in chosen_positions:
                underline = "_" * max(5, len(original_word))
                out_tokens[tok_idx] = f"({next_blank_num}){underline}"
                answer_map[next_blank_num] = original_word
                next_blank_num += 1

        question_paragraphs.append(assemble_tokens(out_tokens))

    word_bank = list(answer_map.values())
    random.shuffle(word_bank)

    return question_paragraphs, answer_map, original_paragraphs, word_bank


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
빈칸에 들어갈 단어는 **단어은행**에서 선택할 수 있으며, 전체 빈칸 수는 **최대 25개**로 제한됩니다.
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

random_seed = st.text_input(
    "랜덤 seed 선택 사항",
    value="",
    placeholder="같은 문제를 다시 만들고 싶을 때 숫자 입력",
)

st.info("빈칸 수는 자동 생성되며 최대 25개로 고정됩니다.")

st.info("답안 입력 방식: 빈칸별 선택")
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

                questions, answer_map, original_paragraphs, word_bank = generate_questions_from_docx(
                    uploaded_file,
                    random_seed=random_seed,
                    max_blanks=25,
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

            except Exception as e:
                st.error("문제 생성 중 오류가 발생했습니다.")
                st.exception(e)

st.markdown("---")


with st.sidebar:
    st.header("📝 문제지")

    if "questions" in st.session_state:
        for para in st.session_state["questions"]:
            if para.strip():
                st.markdown(para)
            else:
                st.write("")
    else:
        st.caption("문제지가 여기에 표시됩니다.")


if "answer_map" not in st.session_state:
    st.info("문제지를 먼저 생성해 주세요.")
else:
    answer_map = st.session_state["answer_map"]
    word_bank = st.session_state.get("word_bank", [])

    if len(answer_map) == 0:
        st.warning("생성된 빈칸이 없습니다. 빈칸 비율을 올리거나 다른 지문을 사용해 보세요.")
    else:
        st.header("📄 문제지")

        for para in st.session_state["questions"]:
            if para.strip():
                st.markdown(para)
            else:
                st.write("")

        st.markdown("---")

        st.subheader("🏦 단어은행")
        st.markdown(" / ".join([f"`{word}`" for word in word_bank]))

        st.markdown("---")

        st.subheader("✏️ 답안 입력")

        for num in sorted(answer_map.keys()):
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
