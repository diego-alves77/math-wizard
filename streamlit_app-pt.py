
import random
import time
from typing import List, Optional
import streamlit as st

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Rapid Number Forge (RNF)", layout="centered")

# ---------------- STATE ----------------
def now_ms():
    return int(time.time() * 1000)

def init_state():
    if "active_level" not in st.session_state:
        st.session_state.active_level = 1
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = 1.0
    if "current" not in st.session_state:
        st.session_state.current = None
    if "q_started_ms" not in st.session_state:
        st.session_state.q_started_ms = None
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "question_nonce" not in st.session_state:
        st.session_state.question_nonce = 0
    if "pending_generate" not in st.session_state:
        st.session_state.pending_generate = False
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

# ---------------- LEVELS ----------------
LEVELS = {
    1: "Atomic",
    2: "Two-Step",
    3: "Running Total",
    4: "Multiply",
    5: "Compression",
}

# ---------------- QUESTION GEN ----------------
def generate_question(level: int):
    rng = random.Random()

    if level == 1:
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        op = rng.choice(["+", "-"])
        expr = f"{a} {op} {b}"
        ans = eval(expr)
        return expr, ans

    if level == 2:
        a = rng.randint(10, 99)
        b = rng.randint(1, 9)
        expr = f"{a} + {b}"
        return expr, a + b

    if level == 3:
        nums = [rng.randint(1, 9) for _ in range(5)]
        total = nums[0]
        expr = str(nums[0])
        for n in nums[1:]:
            expr += f" + {n}"
            total += n
        return expr, total

    if level == 4:
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        expr = f"{a} × {b}"
        return expr, a * b

    # level 5
    a = rng.randint(10, 50)
    b = rng.randint(2, 9)
    expr = f"{a} × {b} - {b}"
    return expr, a * b - b


# ---------------- ONE TAP ----------------
def make_choices(correct: int, n: int) -> List[int]:
    rng = random.Random()
    choices = {correct}
    while len(choices) < n:
        choices.add(correct + rng.randint(-10, 10))
    out = list(choices)
    rng.shuffle(out)
    return out[:n]

def one_tap_panel(choices: List[int]) -> Optional[int]:
    cols = st.columns(2)
    picked = None
    nonce = st.session_state.question_nonce
    for i, val in enumerate(choices):
        with cols[i % 2]:
            if st.button(str(val), use_container_width=True, key=f"choice_{nonce}_{i}"):
                picked = val
    return picked

# ---------------- MAIN ----------------
init_state()

st.title("Rapid Number Forge (RNF)")
st.caption("Treino de aritmética mental")

# Level selector
st.subheader("Selecionar Nível")
st.session_state.active_level = st.selectbox(
    "Nível",
    options=list(LEVELS.keys()),
    format_func=lambda x: f"{x} — {LEVELS[x]}",
    index=st.session_state.active_level - 1,
)

# Start button
if not st.session_state.game_started:
    if st.button("Iniciar Jogo", type="primary"):
        st.session_state.game_started = True
        st.session_state.pending_generate = True
        st.rerun()

# Generate question only after start
if st.session_state.game_started:
    if st.session_state.current is None or st.session_state.pending_generate:
        st.session_state.current = generate_question(st.session_state.active_level)
        st.session_state.q_started_ms = now_ms()
        st.session_state.question_nonce += 1
        st.session_state.pending_generate = False

    expr, correct = st.session_state.current

    st.divider()
    st.subheader("Resolva")
    st.markdown(f"## {expr}")

    choices = make_choices(correct, 4)
    picked = one_tap_panel(choices)

    if picked is not None:
        rt = (now_ms() - st.session_state.q_started_ms) / 1000.0
        st.session_state.attempts += 1

        if picked == correct:
            st.success(f"{"Correto!"} ({rt:.2f}s)")
            st.session_state.score += 1
        else:
            st.error(f"{"Errado. Resposta:"} {correct}")

        st.session_state.current = None
        st.session_state.pending_generate = True
        st.rerun()

# Score display
st.divider()
st.metric("Pontuação", st.session_state.score)
st.metric("Tentativas", st.session_state.attempts)
