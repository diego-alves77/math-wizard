import streamlit as st
import random
import time
from collections import deque
from statistics import mean, median

WINDOW = 10  # rolling accuracy window
N_OPTIONS = 6  # number of answer options to show (4 or 6 are good)

LEVELS = {
    1: "Fatos Atômicos",
    2: "Dois Passos (3 operandos)",
    3: "Total Corrente (total em sequência)",
    4: "Multiplicação",
    5: "Multioperação (3–6 operandos)",
}

ATOMIC_MODES = {
    "Soma": "add",
    "Subtração": "sub",
    "Multiplicação": "mul",
    "Divisão exata": "div_exact",
    "Divisão com resto (só quociente)": "div_quot",
    "Mista": "mixed",
}

MULTIOP_MODES = {
    "Soma": "sum",
    "Subtração": "sub",
    "Soma–Subtração": "sumsub",
    "Multiplicação": "mul",
}


def init_state():
    ss = st.session_state
    ss.setdefault("started", False)
    ss.setdefault("current_problem", None)  # dict with prompt, correct, meta
    ss.setdefault("start_time", None)
    ss.setdefault("history", [])  # list of dicts
    ss.setdefault("rolling_correct", deque(maxlen=WINDOW))
    ss.setdefault("rolling_scores", [])
    ss.setdefault("best_rolling", 0.0)
    ss.setdefault("last_prompt", None)
    ss.setdefault("last_rt", None)
    ss.setdefault("q_id", 0)  # increases each new question (helps reset widgets)


init_state()


# ---------- Utility: errors / distractors ----------

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


def make_options(correct: int, kind: str = "generic", a=None, b=None):
    """
    Create multiple-choice options with plausible distractors.
    kind can be: generic, div_quot, mul, addsub, running
    """
    opts = {correct}

    # Common small-off errors
    for d in (-1, 1, -2, 2, -10, 10):
        opts.add(correct + d)

    # Digit reversal (if reasonable)
    s = str(abs(correct))
    if len(s) >= 2:
        rev = int(s[::-1])
        opts.add(rev if correct >= 0 else -rev)

    # Kind-specific distractors
    if kind == "div_quot" and a is not None and b is not None:
        # remainder confusion or rounding-ish
        opts.add((a + b - 1) // b)  # ceil quotient
        if b != 0:
            opts.add(round(a / b))

    if kind == "mul" and a is not None and b is not None:
        # add instead of multiply / multiply off-by-one operand
        opts.add(a + b)
        opts.add(a * (b + 1))
        opts.add((a + 1) * b)

    if kind in ("addsub", "running"):
        # sign flip
        opts.add(-correct)

    # Ensure we have enough distinct options
    # Fill with random near values
    spread = max(12, abs(correct) // 5 + 12)
    while len(opts) < N_OPTIONS:
        opts.add(correct + random.randint(-spread, spread))

    # If too many, sample but keep correct
    opts = list(opts)
    opts.remove(correct)
    random.shuffle(opts)
    opts = [correct] + opts[: (N_OPTIONS - 1)]
    random.shuffle(opts)
    return opts


def avoid_repeat(prompt: str) -> bool:
    if prompt == st.session_state.last_prompt:
        return False
    st.session_state.last_prompt = prompt
    return True


# ---------- Problem generators ----------

def gen_atomic(mode_key: str):
    mode = ATOMIC_MODES[mode_key]

    while True:
        if mode == "mixed":
            mode = random.choice(["add", "sub", "mul", "div_exact", "div_quot"])

        if mode == "add":
            a, b = random.randint(2, 99), random.randint(2, 99)
            prompt = f"{a} + {b}"
            correct = a + b
            kind = "addsub"

        elif mode == "sub":
            a, b = random.randint(2, 99), random.randint(2, 99)
            # Keep it mostly non-negative for speed drills
            if b > a:
                a, b = b, a
            prompt = f"{a} − {b}"
            correct = a - b
            kind = "addsub"

        elif mode == "mul":
            a, b = random.randint(2, 12), random.randint(2, 12)
            prompt = f"{a} × {b}"
            correct = a * b
            kind = "mul"

        elif mode == "div_exact":
            b = random.randint(2, 12)
            q = random.randint(2, 12)
            a = b * q
            prompt = f"{a} ÷ {b}  (exata)"
            correct = q
            kind = "div_quot"

        elif mode == "div_quot":
            b = random.randint(2, 12)
            q = random.randint(2, 20)
            r = random.randint(1, b - 1)
            a = b * q + r
            prompt = f"{a} ÷ {b}  (só quociente)"
            correct = a // b
            kind = "div_quot"

        else:
            a, b = random.randint(2, 99), random.randint(2, 99)
            prompt = f"{a} + {b}"
            correct = a + b
            kind = "generic"

        if avoid_repeat(prompt):
            break

    options = make_options(correct, kind=kind, a=a, b=b)
    return {
        "prompt": prompt,
        "correct": correct,
        "options": options,
        "tag": f"Fatos Atômicos — {mode_key}",
    }


def gen_two_steps():
    # Explicit parentheses to remove precedence ambiguity:
    # ((a op1 b) op2 c)
    while True:
        a = random.randint(5, 99)
        b = random.randint(2, 30)
        c = random.randint(2, 30)

        op1 = random.choice(["+", "−", "×"])
        op2 = random.choice(["+", "−", "×"])

        def apply(x, op, y):
            if op == "+":
                return x + y
            if op == "−":
                return x - y
            return x * y

        first = apply(a, op1, b)
        correct = apply(first, op2, c)

        prompt = f"({a} {op1} {b}) {op2} {c}"
        if avoid_repeat(prompt):
            break

    options = make_options(correct, kind="addsub")
    return {
        "prompt": prompt,
        "correct": correct,
        "options": options,
        "tag": "Dois Passos — 3 operandos",
    }


def gen_total_corrente():
    # Running total: start value + sequence of +/- steps, compute final total
    # This is what "Total Corrente" most naturally means as a drill.
    while True:
        start = random.randint(10, 99)
        n_steps = random.randint(4, 7)
        steps = []
        total = start
        for _ in range(n_steps):
            op = random.choice(["+", "−"])
            val = random.randint(5, 30)
            steps.append((op, val))
            total = total + val if op == "+" else total - val

        # display: "Comece em 37: +12, −9, +15, ..."
        seq = ", ".join([f"{op}{val}" for op, val in steps])
        prompt = f"Comece em {start}: {seq}  → total final?"
        correct = total

        if avoid_repeat(prompt):
            break

    options = make_options(correct, kind="running")
    return {
        "prompt": prompt,
        "correct": correct,
        "options": options,
        "tag": "Total Corrente — sequência",
    }


def gen_multiplicacao():
    # A dedicated multiplication level, slightly harder than atomic mul
    while True:
        a = random.randint(12, 99)
        b = random.randint(7, 19)
        prompt = f"{a} × {b}"
        correct = a * b
        if avoid_repeat(prompt):
            break

    options = make_options(correct, kind="mul", a=a, b=b)
    return {
        "prompt": prompt,
        "correct": correct,
        "options": options,
        "tag": "Multiplicação — foco",
    }


def gen_multiop(mode_key: str, n_operands: int):
    mode = MULTIOP_MODES[mode_key]

    while True:
        nums = [random.randint(2, 20) for _ in range(n_operands)]

        if mode == "sum":
            expr = " + ".join(map(str, nums))
            correct = sum(nums)
            prompt = expr
            kind = "addsub"

        elif mode == "sub":
            # a - b - c - ...
            # Keep result moderately stable: make first bigger
            nums[0] = random.randint(30, 80)
            expr = " − ".join(map(str, nums))
            correct = nums[0]
            for x in nums[1:]:
                correct -= x
            prompt = expr
            kind = "addsub"

        elif mode == "sumsub":
            # a ± b ± c ...
            ops = [random.choice(["+", "−"]) for _ in range(n_operands - 1)]
            pieces = [str(nums[0])]
            correct = nums[0]
            for op, x in zip(ops, nums[1:]):
                pieces.append(op)
                pieces.append(str(x))
                correct = correct + x if op == "+" else correct - x
            prompt = " ".join(pieces).replace("+", "+").replace("−", "−")
            kind = "addsub"

        else:  # mul
            # Keep products within reasonable range
            nums = [random.randint(2, 9) for _ in range(n_operands)]
            expr = " × ".join(map(str, nums))
            correct = 1
            for x in nums:
                correct *= x
            prompt = expr
            kind = "mul"

        prompt = f"{prompt}  ({n_operands} operandos)"
        if avoid_repeat(prompt):
            break

    options = make_options(correct, kind=kind)
    return {
        "prompt": prompt,
        "correct": correct,
        "options": options,
        "tag": f"Multioperação — {mode_key} — {n_operands} operandos",
    }


def generate_problem(level: int, atomic_mode: str, multi_mode: str, n_operands: int):
    if level == 1:
        return gen_atomic(atomic_mode)
    if level == 2:
        return gen_two_steps()
    if level == 3:
        return gen_total_corrente()
    if level == 4:
        return gen_multiplicacao()
    return gen_multiop(multi_mode, n_operands)


# ---------- UI ----------

st.title("Rapid Number Forge — PT (opções)")

level = st.selectbox(
    "Nível",
    list(LEVELS.keys()),
    format_func=lambda x: f"{x} — {LEVELS[x]}",
)

# Per-level controls
atomic_mode = None
multi_mode = None
n_operands = 3

if level == 1:
    atomic_mode = st.selectbox("Modo (Fatos Atômicos)", list(ATOMIC_MODES.keys()))
elif level == 5:
    c1, c2 = st.columns([2, 1])
    with c1:
        multi_mode = st.selectbox("Tipo (Multioperação)", list(MULTIOP_MODES.keys()))
    with c2:
        n_operands = st.selectbox("Operandos", [3, 4, 5, 6], index=0)

st.divider()

# Start / Reset
colA, colB = st.columns([1, 1])
with colA:
    if not st.session_state.started:
        if st.button("▶️ Iniciar", type="primary"):
            st.session_state.started = True
            st.session_state.current_problem = None
            st.session_state.q_id += 1
            st.rerun()
    else:
        if st.button("⏹️ Reiniciar", type="secondary"):
            for k in [
                "started", "current_problem", "start_time", "history",
                "rolling_correct", "rolling_scores", "best_rolling",
                "last_prompt", "last_rt", "q_id"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            init_state()
            st.rerun()

with colB:
    st.caption(f"Janela rolante: {WINDOW} | Opções por questão: {N_OPTIONS}")

if not st.session_state.started:
    st.stop()

# Create next problem
if st.session_state.current_problem is None:
    st.session_state.current_problem = generate_problem(
        level=level,
        atomic_mode=atomic_mode or "Soma",
        multi_mode=multi_mode or "Soma",
        n_operands=n_operands,
    )
    st.session_state.start_time = time.time()
    st.session_state.q_id += 1

prob = st.session_state.current_problem
prompt = prob["prompt"]
correct = prob["correct"]
options = prob["options"]
tag = prob["tag"]

st.caption(tag)
st.subheader(prompt)

# Answer selection (radio + submit)
with st.form(f"answer_form_{st.session_state.q_id}", clear_on_submit=True):
    choice = st.radio("Escolha sua resposta", options, index=None, horizontal=True)
    submitted = st.form_submit_button("Enviar", type="primary")

if submitted:
    rt = time.time() - st.session_state.start_time
    st.session_state.last_rt = rt

    user = choice if isinstance(choice, int) else None
    is_correct = (user == correct)

    st.session_state.history.append(
        {
            "tag": tag,
            "conta": prompt,
            "entrada": user,
            "correto": correct,
            "rt_s": rt,
            "erro": detect_error(user, correct),
        }
    )

    st.session_state.rolling_correct.append(is_correct)

    if len(st.session_state.rolling_correct) == WINDOW:
        pct = 100.0 * sum(st.session_state.rolling_correct) / WINDOW
        st.session_state.rolling_scores.append(pct)
        st.session_state.best_rolling = max(st.session_state.best_rolling, pct)

    st.session_state.current_problem = None
    st.rerun()

st.divider()

# Stats
total = len(st.session_state.history)
correct_total = sum(1 for r in st.session_state.history if r["erro"] == "correto")
acc_total = (100.0 * correct_total / total) if total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Respostas", f"{total}")
c2.metric("Acurácia total", f"{acc_total:.1f}%")
c3.metric("Melhor rolante", f"{st.session_state.best_rolling:.1f}%")
if st.session_state.last_rt is None:
    c4.metric("Último tempo", "—")
else:
    c4.metric("Último tempo", f"{st.session_state.last_rt:.2f}s")

if total:
    rts = [r["rt_s"] for r in st.session_state.history]
    st.caption(f"RT média: {mean(rts):.2f}s | mediana: {median(rts):.2f}s")

# Rolling chart
if len(st.session_state.rolling_correct) < WINDOW:
    st.info(f"Acurácia rolante aparece após {WINDOW} respostas.")
else:
    current_pct = st.session_state.rolling_scores[-1]
    st.metric(f"Acurácia rolante (últimas {WINDOW})", f"{current_pct:.1f}%")
    st.line_chart(st.session_state.rolling_scores)

# Recent history
if total:
    st.subheader("Histórico recente")
    st.dataframe(st.session_state.history[-20:], use_container_width=True)
