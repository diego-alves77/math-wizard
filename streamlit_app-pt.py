import streamlit as st
import random
import time
from collections import deque
from statistics import mean, median

WINDOW = 10  # rolling window size

LEVELS = {
    1: "Fatos Atômicos",
    2: "Dois Passos",
    3: "Total Corrente",
    4: "Multiplicação",
    5: "Compressão",
}


def init_state():
    ss = st.session_state
    ss.setdefault("started", False)
    ss.setdefault("current_problem", None)      # tuple(prompt, correct)
    ss.setdefault("start_time", None)
    ss.setdefault("history", [])                # list of dicts
    ss.setdefault("rolling_correct", deque(maxlen=WINDOW))  # bools
    ss.setdefault("rolling_scores", [])         # list of rolling accuracies (%)
    ss.setdefault("best_rolling", 0.0)
    ss.setdefault("last_prompt", None)


init_state()


def generate_problem(level: int):
    # Avoid repeating the immediate last prompt
    while True:
        if level == 1:
            a, b = random.randint(2, 9), random.randint(2, 9)
            op = random.choice(["+", "-", "×"])
        elif level == 2:
            a, b = random.randint(10, 99), random.randint(2, 9)
            op = random.choice(["+", "-", "×"])
        elif level == 3:
            a, b = random.randint(20, 99), random.randint(10, 50)
            op = random.choice(["+", "-"])
        elif level == 4:
            a, b = random.randint(12, 40), random.randint(2, 12)
            op = "×"
        else:
            a, b = random.randint(50, 200), random.randint(10, 50)
            op = random.choice(["+", "-", "×"])

        prompt = f"{a} {op} {b}"
        if prompt != st.session_state.last_prompt:
            st.session_state.last_prompt = prompt
            break

    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    else:
        ans = a * b

    return prompt, ans


def detect_error(user, correct):
    if user is None:
        return "sem resposta"
    if user == correct:
        return "correto"
    if abs(user - correct) == 1:
        return "erro de ±1"
    if str(user)[::-1] == str(correct):
        return "inversão de dígitos"
    if (user < 0) != (correct < 0):
        return "erro de sinal"
    return "erro de cálculo"


def format_math(prompt: str) -> str:
    a_str, op, b_str = prompt.split()
    width = max(len(a_str), len(b_str)) + 2  # +2 to look nicer
    top = a_str.rjust(width)
    bot = f"{op} {b_str.rjust(width - 2)}"
    line = "-" * width
    return f"{top}\n{bot}\n{line}"


st.title("Rapid Number Forge — PT")

level = st.selectbox(
    "Nível",
    list(LEVELS.keys()),
    format_func=lambda x: f"{x} — {LEVELS[x]}",
)

st.divider()

colA, colB = st.columns([1, 1])
with colA:
    if not st.session_state.started:
        if st.button("▶️ Iniciar Jogo", type="primary"):
            st.session_state.started = True
            st.session_state.current_problem = None
            st.rerun()
    else:
        if st.button("⏹️ Reiniciar", type="secondary"):
            # Full reset of the game state (but keep UI selections)
            for k in [
                "started", "current_problem", "start_time", "history",
                "rolling_correct", "rolling_scores", "best_rolling", "last_prompt"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            init_state()
            st.rerun()

with colB:
    st.caption(f"Janela rolante: {WINDOW} respostas")

if not st.session_state.started:
    st.stop()

# Create next problem if needed
if st.session_state.current_problem is None:
    st.session_state.current_problem = generate_problem(level)
    st.session_state.start_time = time.time()

prompt, correct = st.session_state.current_problem
st.code(format_math(prompt))

# Use a form so Enter submits and input clears
with st.form("answer_form", clear_on_submit=True):
    answer = st.text_input("Sua resposta", placeholder="Digite um inteiro e pressione Enter")
    submitted = st.form_submit_button("Enviar", type="primary")

if submitted:
    try:
        user = int(answer.strip())
    except Exception:
        user = None

    rt = time.time() - st.session_state.start_time
    is_correct = (user == correct)

    st.session_state.history.append(
        {
            "conta": prompt,
            "entrada": user,
            "correto": correct,
            "rt_s": rt,
            "erro": detect_error(user, correct),
        }
    )

    st.session_state.rolling_correct.append(is_correct)

    # Update rolling score only when the window is "full"
    if len(st.session_state.rolling_correct) == WINDOW:
        pct = 100.0 * sum(st.session_state.rolling_correct) / WINDOW
        st.session_state.rolling_scores.append(pct)
        st.session_state.best_rolling = max(st.session_state.best_rolling, pct)

    # Next problem
    st.session_state.current_problem = None
    st.rerun()

st.divider()

# Stats panel
total = len(st.session_state.history)
correct_total = sum(1 for r in st.session_state.history if r["erro"] == "correto")
acc_total = (100.0 * correct_total / total) if total else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Respostas", f"{total}")
col2.metric("Acurácia total", f"{acc_total:.1f}%")
col3.metric("Melhor rolante", f"{st.session_state.best_rolling:.1f}%")

# Reaction time stats
if total:
    rts = [r["rt_s"] for r in st.session_state.history]
    st.caption(f"Tempo de resposta — média: {mean(rts):.2f}s | mediana: {median(rts):.2f}s")

# Rolling chart
if len(st.session_state.rolling_correct) < WINDOW:
    st.info(f"Acurácia rolante aparece após {WINDOW} respostas.")
else:
    current_pct = st.session_state.rolling_scores[-1]
    st.metric(f"Acurácia rolante (últimas {WINDOW})", f"{current_pct:.1f}%")
    st.line_chart(st.session_state.rolling_scores)

# Report
if len(st.session_state.rolling_scores) >= 10:
    st.divider()
    st.subheader("Relatório (últimas 10 janelas rolantes)")
    # last 10 windows = last 10 scores; show the last 10*WINDOW answers makes sense contextually
    last_answers = st.session_state.history[-(10 * WINDOW):]
    st.dataframe(last_answers, use_container_width=True)
