
import streamlit as st
import random
import time
from collections import deque
from statistics import mean, median

# =========================
# CONFIG
# =========================
WINDOW = 10
N_OPTIONS = 3  # always exactly 3 answer buttons
GAME_TITLE = "Mago da Matemática"

LEVELS = {
    1: "Conta Simples",
    2: "Conta Dupla",
    3: "Memória",
}

# Progression inside Conta Simples (must be completed in this order)
ATOMIC_MODES = [
    "Soma",
    "Subtração",
    "Multiplicação",
    "Divisão exata",
    "Divisão com resto (só quociente)",
    "Mista",
]

ATOMIC_MODE_TO_CODE = {
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

# Intermediate thresholds (seconds) to allow "advance" (with accuracy >= 90%)
TARGETS_ATOMIC_INTERMEDIATE_HI = {
    "Soma": 2.0,
    "Subtração": 2.0,
    "Multiplicação": 2.5,
    "Divisão exata": 3.0,
    "Divisão com resto (só quociente)": 3.5,
    "Mista": 2.5,
}

TARGET_TWO_STEPS_INTERMEDIATE_HI = 4.0

TARGETS_MULTIOP_INTERMEDIATE_HI = {
    3: 4.5,
    4: 6.0,
    5: 8.0,
    6: 10.0,
}

# =========================
# STATE
# =========================
def init_state():
    ss = st.session_state
    ss.setdefault("started", False)
    ss.setdefault("current_problem", None)
    ss.setdefault("start_time", None)

    ss.setdefault("history", [])
    ss.setdefault("rolling_correct", deque(maxlen=WINDOW))
    ss.setdefault("rolling_scores", [])
    ss.setdefault("best_rolling", 0.0)

    ss.setdefault("last_prompt_key", None)
    ss.setdefault("last_rt", None)
    ss.setdefault("q_id", 0)

    # widget keys
    ss.setdefault("level_choice", 1)
    ss.setdefault("atomic_mode_choice", ATOMIC_MODES[0])
    ss.setdefault("multi_mode_choice", "Soma")
    ss.setdefault("n_operands_choice", 3)


init_state()

# =========================
# NAVIGATION (callbacks)
# Progression: Conta Simples (each mode) -> Conta Dupla -> Memória
# Menu still allows jumping anywhere.
# =========================
def _next_stage(level: int, atomic_mode: str):
    level = int(level)
    if level == 1:
        idx = ATOMIC_MODES.index(atomic_mode)
        if idx < len(ATOMIC_MODES) - 1:
            return 1, ATOMIC_MODES[idx + 1]
        return 2, None
    if level == 2:
        return 3, None
    return 3, None


def _prev_stage(level: int, atomic_mode: str):
    level = int(level)
    if level == 1:
        idx = ATOMIC_MODES.index(atomic_mode)
        if idx > 0:
            return 1, ATOMIC_MODES[idx - 1]
        return 1, ATOMIC_MODES[0]
    if level == 2:
        return 1, ATOMIC_MODES[-1]
    return 2, None


def goto_stage(direction: str, level: int, atomic_mode: str):
    if direction == "next":
        new_level, new_atomic = _next_stage(level, atomic_mode)
    else:
        new_level, new_atomic = _prev_stage(level, atomic_mode)

    st.session_state["level_choice"] = int(new_level)
    if int(new_level) == 1 and new_atomic is not None:
        st.session_state["atomic_mode_choice"] = new_atomic

    st.session_state["current_problem"] = None
    st.session_state["q_id"] += 1


def goto_prev(level: int, atomic_mode: str):
    goto_stage("prev", level, atomic_mode)


def goto_next(level: int, atomic_mode: str):
    goto_stage("next", level, atomic_mode)


# =========================
# METRICS + HELPERS
# =========================
def rolling_accuracy_percent():
    if len(st.session_state.rolling_correct) < WINDOW:
        return None
    return 100.0 * sum(st.session_state.rolling_correct) / WINDOW


def rt_mean_median():
    if not st.session_state.history:
        return None, None
    rts = [r["rt_s"] for r in st.session_state.history]
    return mean(rts), median(rts)


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
    opts = {correct}

    # near misses
    for d in (-1, 1, -2, 2, -10, 10):
        opts.add(correct + d)

    # digit reversal
    s = str(abs(correct))
    if len(s) >= 2:
        rev = int(s[::-1])
        opts.add(rev if correct >= 0 else -rev)

    # operation-specific distractors
    if kind == "mul" and a is not None and b is not None:
        opts.add(a + b)
        opts.add(a * (b + 1))
        opts.add((a + 1) * b)

    if kind == "div_quot" and a is not None and b is not None and b != 0:
        opts.add(round(a / b))
        opts.add((a + b - 1) // b)

    if kind == "addsub":
        opts.add(-correct)

    spread = max(12, abs(correct) // 5 + 12)
    while len(opts) < N_OPTIONS:
        opts.add(correct + random.randint(-spread, spread))

    # choose exactly N_OPTIONS, always include correct
    opts = list(opts)
    if correct in opts:
        opts.remove(correct)
    random.shuffle(opts)
    selected = [correct] + opts[: N_OPTIONS - 1]
    random.shuffle(selected)
    return selected


def avoid_repeat(prompt_key: str) -> bool:
    if prompt_key == st.session_state.last_prompt_key:
        return False
    st.session_state.last_prompt_key = prompt_key
    return True


def render_vertical_expression(nums, ops):
    """
    One operand per line, aligned INCLUDING the first line.
    First line gets a blank operator slot ("  ") so it aligns with "+ ", "− ", "× ", "÷ ".
    """
    width = max(len(str(n)) for n in nums)
    lines = [f"  {str(nums[0]).rjust(width)}"]
    for op, n in zip(ops, nums[1:]):
        lines.append(f"{op} {str(n).rjust(width)}")
    lines.append("-" * (width + 2))
    return "\n".join(lines)


def memory_kb_for_problem(nums, ops) -> float:
    """
    "Carga de Memória": bytes UTF-8 dos operandos + operadores (inclui o operador do desafio),
    convertidos para kB (1 kB = 1024 bytes).
    """
    tokens = [str(n) for n in nums] + [str(o) for o in ops]
    total_bytes = sum(len(t.encode("utf-8")) for t in tokens)
    return total_bytes / 1024.0


# =========================
# PROBLEM GENERATORS
# =========================
def gen_atomic(mode_key: str):
    mode = ATOMIC_MODE_TO_CODE[mode_key]

    while True:
        chosen = mode
        if chosen == "mixed":
            chosen = random.choice(["add", "sub", "mul", "div_exact", "div_quot"])

        if chosen == "add":
            a, b = random.randint(2, 99), random.randint(2, 99)
            display = f"{a} + {b}"
            correct = a + b
            kind = "addsub"
            prompt_key = display
            nums = [a, b]
            ops = ["+"]

        elif chosen == "sub":
            a, b = random.randint(2, 99), random.randint(2, 99)
            if b > a:
                a, b = b, a
            display = f"{a} − {b}"
            correct = a - b
            kind = "addsub"
            prompt_key = display
            nums = [a, b]
            ops = ["−"]

        elif chosen == "mul":
            a, b = random.randint(2, 12), random.randint(2, 12)
            display = f"{a} × {b}"
            correct = a * b
            kind = "mul"
            prompt_key = display
            nums = [a, b]
            ops = ["×"]

        elif chosen == "div_exact":
            b = random.randint(2, 12)
            q = random.randint(2, 12)
            a = b * q
            display = f"{a} ÷ {b}"
            correct = q
            kind = "div_quot"
            prompt_key = display
            nums = [a, b]
            ops = ["÷"]

        elif chosen == "div_quot":
            b = random.randint(2, 12)
            q = random.randint(2, 20)
            r = random.randint(1, b - 1)
            a = b * q + r
            display = f"{a} ÷ {b}"
            correct = a // b
            kind = "div_quot"
            prompt_key = f"{display}|q"
            nums = [a, b]
            ops = ["÷"]

        else:
            a, b = random.randint(2, 99), random.randint(2, 99)
            display = f"{a} + {b}"
            correct = a + b
            kind = "generic"
            prompt_key = display
            nums = [a, b]
            ops = ["+"]

        if avoid_repeat(prompt_key):
            break

    options = make_options(correct, kind=kind, a=a, b=b)
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": f"Conta Simples — {mode_key}",
        "kind": "inline",
        "nums": nums,
        "ops": ops,
    }


def gen_two_steps():
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
        display = f"({a} {op1} {b}) {op2} {c}"

        if avoid_repeat(display):
            break

    options = make_options(correct, kind="addsub")
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": "Conta Dupla",
        "kind": "inline",
        "nums": [a, b, c],
        "ops": [op1, op2],
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

        prompt_key = f"{mode_key}:{n_operands}:{','.join(map(str, nums))}:{''.join(ops)}"
        if avoid_repeat(prompt_key):
            break

    display = render_vertical_expression(nums, ops)
    options = make_options(correct, kind=kind)
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": f"Memória — {mode_key}",
        "kind": "vertical",
        "nums": nums,
        "ops": ops,
    }


def generate_problem(level: int, atomic_mode: str, multi_mode: str, n_operands: int):
    if level == 1:
        return gen_atomic(atomic_mode)
    if level == 2:
        return gen_two_steps()
    return gen_multiop(multi_mode, n_operands)


# =========================
# END-OF-PAGE REFERENCE + GUIDANCE
# =========================
def targets_text_for_context(level, atomic_mode=None, n_operands=None):
    if level == 1:
        if atomic_mode in ["Soma", "Subtração"]:
            return "Meta de RT: Iniciante 2,0–3,0 s | Intermediário 1,2–2,0 s | Fluência alta < 1,0 s"
        if atomic_mode == "Multiplicação":
            return "Meta de RT: Iniciante 2,5–4,0 s | Intermediário 1,5–2,5 s | Fluência alta < 1,2 s"
        if atomic_mode == "Divisão exata":
            return "Meta de RT: Iniciante 3,0–4,5 s | Intermediário 2,0–3,0 s | Fluência alta < 1,8 s"
        if atomic_mode == "Divisão com resto (só quociente)":
            return "Meta de RT: Iniciante 3,5–5,0 s | Intermediário 2,5–3,5 s | Fluência alta < 2,0 s"
        return "Modo misto: use as metas dos submodos como referência."
    if level == 2:
        return "Meta de RT: Iniciante 4–6 s | Intermediário 3–4 s | Avançado < 3 s"
    # level == 3
    if int(n_operands) == 3:
        return "Meta de RT: Iniciante 4–7 s | Intermediário 3–4,5 s | Avançado < 3 s"
    if int(n_operands) == 4:
        return "Meta de RT: Iniciante 6–9 s | Intermediário 4–6 s | Avançado < 4 s"
    if int(n_operands) == 5:
        return "Meta de RT: Iniciante 8–12 s | Intermediário 6–8 s | Avançado < 6 s"
    return "Meta de RT: Iniciante 10–15 s | Intermediário 8–10 s | Avançado < 7 s"


def recommend_progress(level, atomic_mode, n_operands):
    total = len(st.session_state.history)
    if total == 0:
        return "Responda algumas questões para eu recomendar com base em dados.", False

    roll = rolling_accuracy_percent()
    correct_total = sum(1 for r in st.session_state.history if r["erro"] == "correto")
    acc_total = 100.0 * correct_total / total
    acc_used = roll if roll is not None else acc_total
    acc_label = "acurácia rolante" if roll is not None else "acurácia total"

    _, rt_med = rt_mean_median()
    if rt_med is None:
        return "Sem dados de tempo suficientes para recomendar.", False

    if level == 1:
        thr = TARGETS_ATOMIC_INTERMEDIATE_HI.get(atomic_mode, 2.5)
        ctx = f"Conta Simples ({atomic_mode})"
        nxt_level, nxt_atomic = _next_stage(1, atomic_mode)
        next_label = f"{nxt_atomic} (Conta Simples)" if nxt_level == 1 else "Conta Dupla"
    elif level == 2:
        thr = TARGET_TWO_STEPS_INTERMEDIATE_HI
        ctx = "Conta Dupla"
        next_label = "Memória"
    else:
        thr = TARGETS_MULTIOP_INTERMEDIATE_HI.get(int(n_operands), 6.0)
        ctx = f"Memória ({n_operands})"
        next_label = None

    if acc_used < 90.0:
        return (
            f"Recomendação: **treine mais aqui**.\n\n"
            f"Motivo: sua {acc_label} está em **{acc_used:.1f}%** (meta: ≥ 90%)."
        ), False

    if rt_med > thr:
        return (
            f"Recomendação: **treine mais aqui**.\n\n"
            f"Motivo: sua **RT mediana** está em **{rt_med:.2f}s**, acima da meta intermediária "
            f"(≤ {thr:.2f}s) para **{ctx}**."
        ), False

    if next_label is None:
        return (
            f"Recomendação: **continue refinando**.\n\n"
            f"Você está com {acc_label} **{acc_used:.1f}%** e **RT mediana {rt_med:.2f}s** "
            f"(meta intermediária: ≤ {thr:.2f}s)."
        ), True

    return (
        f"Recomendação: **pode avançar**.\n\n"
        f"Você está com {acc_label} **{acc_used:.1f}%** e **RT mediana {rt_med:.2f}s** "
        f"(meta intermediária: ≤ {thr:.2f}s). Agora siga para **{next_label}**."
    ), True


def show_reference_and_navigation(level, atomic_mode_for_path, n_operands, current_mem_kb):
    st.divider()
    st.subheader("Referência e Orientação")

    st.markdown(
        """
**Siglas e termos (nesta página)**  
- **RT** = *Tempo de Resposta* (tempo entre aparecer a conta e você clicar na resposta).  
- **RT média** = média aritmética dos tempos (pode ser distorcida por respostas muito lentas).  
- **RT mediana** = valor central dos tempos ordenados (representa melhor seu ritmo típico).  
- **Acurácia rolante** = porcentagem de acertos nas **últimas 10** respostas.
"""
    )

    st.markdown("### Metas de RT (neste modo)")
    if level == 1:
        st.write(targets_text_for_context(level, atomic_mode=atomic_mode_for_path))
    elif level == 2:
        st.write(targets_text_for_context(level))
    else:
        st.write(targets_text_for_context(level, n_operands=n_operands))
    st.caption("Regra: velocidade só conta se a acurácia rolante estiver em **90% ou mais**.")

    st.markdown("### Carga de Memória")
    st.write(f"**{current_mem_kb:.4f} kB**")
    st.caption(
        "Estimativa: bytes UTF-8 dos **operandos + operadores** (inclui o operador do desafio), "
        "convertidos para kB (1 kB = 1024 bytes)."
    )

    st.markdown("### Próximo passo")
    rec, _ = recommend_progress(level, atomic_mode_for_path, n_operands)
    st.info(rec)

    nav_left, nav_right = st.columns(2)
    with nav_left:
        st.button(
            "⬅ Voltar",
            use_container_width=True,
            disabled=(level == 1 and atomic_mode_for_path == ATOMIC_MODES[0]),
            on_click=goto_prev,
            args=(level, atomic_mode_for_path),
            key=f"nav_prev_{st.session_state.q_id}",
        )
    with nav_right:
        st.button(
            "Avançar ➡",
            use_container_width=True,
            disabled=(level == 3),
            on_click=goto_next,
            args=(level, atomic_mode_for_path),
            key=f"nav_next_{st.session_state.q_id}",
        )


# =========================
# UI
# =========================
st.title(GAME_TITLE)

level = st.selectbox(
    "Opção do jogo",
    list(LEVELS.keys()),
    key="level_choice",
    format_func=lambda x: f"{x} — {LEVELS[x]}",
)

# Keep a valid atomic mode for progression, even if user is on level 2/3
atomic_mode_for_path = st.session_state["atomic_mode_choice"]
multi_mode = st.session_state["multi_mode_choice"]
n_operands = st.session_state["n_operands_choice"]

if level == 1:
    atomic_mode_for_path = st.selectbox(
        "Modo (Conta Simples)",
        ATOMIC_MODES,
        key="atomic_mode_choice",
    )
elif level == 3:
    c1, c2 = st.columns([2, 1])
    with c1:
        multi_mode = st.selectbox(
            "Tipo (Memória)",
            list(MULTIOP_MODES.keys()),
            key="multi_mode_choice",
        )
    with c2:
        n_operands = st.selectbox(
            "Operandos",
            [3, 4, 5, 6],
            key="n_operands_choice",
        )

st.divider()

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
            # reset gameplay state, keep widget keys as-is
            for k in [
                "started",
                "current_problem",
                "start_time",
                "history",
                "rolling_correct",
                "rolling_scores",
                "best_rolling",
                "last_prompt_key",
                "last_rt",
                "q_id",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            init_state()
            st.rerun()

with colB:
    st.caption(f"Janela rolante: {WINDOW}")

if not st.session_state.started:
    st.stop()

# Create next problem
if st.session_state.current_problem is None:
    st.session_state.current_problem = generate_problem(
        level=level,
        atomic_mode=atomic_mode_for_path,
        multi_mode=multi_mode,
        n_operands=n_operands,
    )
    st.session_state.start_time = time.time()
    st.session_state.q_id += 1

prob = st.session_state.current_problem
display = prob["display"]
correct = prob["correct"]
options = prob["options"]
tag = prob["tag"]
kind = prob["kind"]
nums = prob["nums"]
ops = prob["ops"]

current_mem_kb = memory_kb_for_problem(nums, ops)

st.caption(tag)

# Display problem
if kind == "vertical":
    st.code(display)
else:
    st.subheader(display)

# Answer buttons: 3 buttons, 1 line (columns)
btn_cols = st.columns(3)
clicked_value = None
for i, opt in enumerate(options):
    with btn_cols[i]:
        if st.button(
            str(opt),
            key=f"opt_{st.session_state.q_id}_{i}",
            use_container_width=True,
        ):
            clicked_value = opt

# Handle answer click
if clicked_value is not None:
    rt = time.time() - st.session_state.start_time
    st.session_state.last_rt = rt

    is_correct = (clicked_value == correct)

    st.session_state.history.append(
        {
            "tag": tag,
            "entrada": clicked_value,
            "correto": correct,
            "rt_s": rt,
            "erro": detect_error(clicked_value, correct),
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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Respostas", f"{total}")
m2.metric("Acurácia total", f"{acc_total:.1f}%")
m3.metric("Melhor rolante", f"{st.session_state.best_rolling:.1f}%")
m4.metric("Último tempo", "—" if st.session_state.last_rt is None else f"{st.session_state.last_rt:.2f}s")

if total:
    rts = [r["rt_s"] for r in st.session_state.history]
    st.caption(f"RT média: {mean(rts):.2f}s | RT mediana: {median(rts):.2f}s")

# Rolling chart
if len(st.session_state.rolling_correct) < WINDOW:
    st.info(f"Acurácia rolante aparece após {WINDOW} respostas.")
else:
    current_pct = st.session_state.rolling_scores[-1]
    st.metric(f"Acurácia rolante (últimas {WINDOW})", f"{current_pct:.1f}%")
    st.line_chart(st.session_state.rolling_scores)

# History
if total:
    st.subheader("Histórico recente")
    st.dataframe(st.session_state.history[-20:], use_container_width=True)

# End-of-page reference + guidance + navigation + memory kB
show_reference_and_navigation(
    level=level,
    atomic_mode_for_path=atomic_mode_for_path,
    n_operands=n_operands,
    current_mem_kb=current_mem_kb,
)
