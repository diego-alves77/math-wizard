import streamlit as st
import random
import time
from collections import deque
from statistics import mean, median

WINDOW = 10
N_OPTIONS = 3  # exactly 3 buttons

LEVELS = {
    1: "Fatos Atômicos",
    2: "Dois Passos (3 operandos)",
    3: "Multioperação (3–6 operandos)",
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
    ss.setdefault("current_problem", None)  # dict
    ss.setdefault("start_time", None)
    ss.setdefault("history", [])
    ss.setdefault("rolling_correct", deque(maxlen=WINDOW))
    ss.setdefault("rolling_scores", [])
    ss.setdefault("best_rolling", 0.0)
    ss.setdefault("last_prompt", None)
    ss.setdefault("last_rt", None)
    ss.setdefault("q_id", 0)  # new question id


init_state()


# ---------- Error labeling ----------
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


# ---------- Options / distractors ----------
def make_options(correct: int, kind: str = "generic", a=None, b=None):
    opts = {correct}

    # most common near-miss distractors
    for d in (-1, 1, -2, 2, -10, 10):
        opts.add(correct + d)

    # digit reversal (if meaningful)
    s = str(abs(correct))
    if len(s) >= 2:
        rev = int(s[::-1])
        opts.add(rev if correct >= 0 else -rev)

    # operation-specific distractors
    if kind == "mul" and a is not None and b is not None:
        opts.add(a + b)          # add instead of multiply
        opts.add(a * (b + 1))    # off-by-one operand
        opts.add((a + 1) * b)

    if kind == "div_quot" and a is not None and b is not None and b != 0:
        opts.add(round(a / b))
        opts.add((a + b - 1) // b)  # ceil quotient

    if kind == "addsub":
        opts.add(-correct)  # sign flip mistake

    # Ensure >= N_OPTIONS unique
    spread = max(12, abs(correct) // 5 + 12)
    while len(opts) < N_OPTIONS:
        opts.add(correct + random.randint(-spread, spread))

    # Select exactly N_OPTIONS, always include correct
    opts = list(opts)
    opts.remove(correct)
    random.shuffle(opts)
    selected = [correct] + opts[: N_OPTIONS - 1]
    random.shuffle(selected)
    return selected


def avoid_repeat(prompt: str) -> bool:
    if prompt == st.session_state.last_prompt:
        return False
    st.session_state.last_prompt = prompt
    return True


# ---------- Render helpers ----------
def render_vertical_expression(nums, ops):
    """
    nums: list[int]
    ops: list[str] length = len(nums)-1, each '+' '−' '×'
    Render one operand per line, aligned, with operator prefix.
    Example:
      12
    +  7
    −  3
    ×  2
    ----
    """
    width = max(len(str(n)) for n in nums)
    lines = [str(nums[0]).rjust(width)]
    for op, n in zip(ops, nums[1:]):
        lines.append(f"{op} {str(n).rjust(width)}")
    lines.append("-" * (width + 2))
    return "\n".join(lines)


# ---------- Problem generators ----------
def gen_atomic(mode_key: str):
    mode = ATOMIC_MODES[mode_key]

    while True:
        chosen = mode
        if chosen == "mixed":
            chosen = random.choice(["add", "sub", "mul", "div_exact", "div_quot"])

        if chosen == "add":
            a, b = random.randint(2, 99), random.randint(2, 99)
            prompt = f"{a} + {b}"
            correct = a + b
            kind = "addsub"

        elif chosen == "sub":
            a, b = random.randint(2, 99), random.randint(2, 99)
            if b > a:
                a, b = b, a
            prompt = f"{a} − {b}"
            correct = a - b
            kind = "addsub"

        elif chosen == "mul":
            a, b = random.randint(2, 12), random.randint(2, 12)
            prompt = f"{a} × {b}"
            correct = a * b
            kind = "mul"

        elif chosen == "div_exact":
            b = random.randint(2, 12)
            q = random.randint(2, 12)
            a = b * q
            prompt = f"{a} ÷ {b}"
            correct = q
            kind = "div_quot"

        elif chosen == "div_quot":
            b = random.randint(2, 12)
            q = random.randint(2, 20)
            r = random.randint(1, b - 1)
            a = b * q + r
            prompt = f"{a} ÷ {b} (quociente)"
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
    # ((a op1 b) op2 c) with explicit parentheses
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


def gen_multiop(mode_key: str, n_operands: int):
    mode = MULTIOP_MODES[mode_key]

    while True:
        nums = [random.randint(2, 20) for _ in range(n_operands)]

        if mode == "sum":
            ops = ["+"] * (n_operands - 1)
            correct = sum(nums)
            kind = "addsub"

        elif mode == "sub":
            nums[0] = random.randint(30, 80)
            ops = ["−"] * (n_operands - 1)
            correct = nums[0]
            for x in nums[1:]:
                correct -= x
            kind = "addsub"

        elif mode == "sumsub":
            ops = [random.choice(["+", "−"]) for _ in range(n_operands - 1)]
            correct = nums[0]
            for op, x in zip(ops, nums[1:]):
                correct = correct + x if op == "+" else correct - x
            kind = "addsub"

        else:  # mul
            nums = [random.randint(2, 9) for _ in range(n_operands)]
            ops = ["×"] * (n_operands - 1)
            correct = 1
            for x in nums:
                correct *= x
            kind = "mul"

        # prompt rendered vertically (not inline), so prompt is a stable key string
        prompt_key = f"{mode_key}:{n_operands}:{','.join(map(str, nums))}:{''.join(ops)}"
        if avoid_repeat(prompt_key):
            break

    pretty = render_vertical_expression(nums, ops)
    options = make_options(correct, kind=kind)
    return {
        "prompt": pretty,        # for display
        "prompt_key": prompt_key, # internal uniqueness
        "correct": correct,
        "options": options,
        "tag": f"Multioperação — {mode_key}",
    }


def generate_problem(level: int, atomic_mode: str, multi_mode: str, n_operands: int):
    if level == 1:
        return gen_atomic(atomic_mode)
    if level == 2:
        return gen_two_steps()
    return gen_multiop(multi_mode, n_operands)


# ---------- UI ----------
st.title("Rapid Number Forge — PT (botões)")

level = st.selectbox(
    "Nível",
    list(LEVELS.keys()),
    format_func=lambda x: f"{x} — {LEVELS[x]}",
)

atomic_mode = "Soma"
multi_mode = "Soma"
n_operands = 3

if level == 1:
    atomic_mode = st.selectbox("Modo (Fatos Atômicos)", list(ATOMIC_MODES.keys()))
elif level == 3:
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
    st.caption(f"Janela rolante: {WINDOW} | Opções: {N_OPTIONS}")

if not st.session_state.started:
    st.stop()

# Create next problem
if st.session_state.current_problem is None:
    st.session_state.current_problem = generate_problem(
        level=level,
        atomic_mode=atomic_mode,
        multi_mode=multi_mode,
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

# Display problem
if level == 3:
    st.code(prompt)  # already vertical aligned
else:
    st.subheader(prompt)

# Answer buttons: one click = submit
btn_cols = st.columns(N_OPTIONS)
clicked_value = None
for i, opt in enumerate(options):
    with btn_cols[i]:
        if st.button(str(opt), key=f"opt_{st.session_state.q_id}_{i}", use_container_width=True):
            clicked_value = opt

if clicked_value is not None:
    rt = time.time() - st.session_state.start_time
    st.session_state.last_rt = rt

    user = clicked_value
    is_correct = (user == correct)

    st.session_state.history.append(
        {
            "tag": tag,
            "conta": prompt if level != 3 else "(multioperação)",
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
c4.metric("Último tempo", "—" if st.session_state.last_rt is None else f"{st.session_state.last_rt:.2f}s")

if total:
    rts = [r["rt_s"] for r in st.session_state.history]
    st.caption(f"RT média: {mean(rts):.2f}s | mediana: {median(rts):.2f}s")

if len(st.session_state.rolling_correct) < WINDOW:
    st.info(f"Acurácia rolante aparece após {WINDOW} respostas.")
else:
    current_pct = st.session_state.rolling_scores[-1]
    st.metric(f"Acurácia rolante (últimas {WINDOW})", f"{current_pct:.1f}%")
    st.line_chart(st.session_state.rolling_scores)

if total:
    st.subheader("Histórico recente")
    st.dataframe(st.session_state.history[-20:], use_container_width=True)
