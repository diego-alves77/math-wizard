import streamlit as st
import random
import time
from collections import deque
from statistics import mean, median

WINDOW = 10
N_OPTIONS = 3  # fixed: 3 answer buttons

LEVELS = {
    1: "Fatos Atômicos",
    2: "Dois Passos",
    3: "Multioperação",
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


# ---------- Targets (seconds) ----------
# Use these to decide "advance or train more".
TARGETS_ATOMIC = {
    "Soma":            {"beginner_hi": 3.0, "intermediate_hi": 2.0, "advanced_hi": 1.0},
    "Subtração":       {"beginner_hi": 3.0, "intermediate_hi": 2.0, "advanced_hi": 1.0},
    "Multiplicação":   {"beginner_hi": 4.0, "intermediate_hi": 2.5, "advanced_hi": 1.2},
    "Divisão exata":   {"beginner_hi": 4.5, "intermediate_hi": 3.0, "advanced_hi": 1.8},
    "Divisão com resto (só quociente)": {"beginner_hi": 5.0, "intermediate_hi": 3.5, "advanced_hi": 2.0},
    "Mista": None,  # handled specially
}

TARGETS_TWO_STEPS = {"beginner_hi": 6.0, "intermediate_hi": 4.0, "advanced_hi": 3.0}

TARGETS_MULTIOP = {
    3: {"beginner_hi": 7.0, "intermediate_hi": 4.5, "advanced_hi": 3.0},
    4: {"beginner_hi": 9.0, "intermediate_hi": 6.0, "advanced_hi": 4.0},
    5: {"beginner_hi": 12.0, "intermediate_hi": 8.0, "advanced_hi": 6.0},
    6: {"beginner_hi": 15.0, "intermediate_hi": 10.0, "advanced_hi": 7.0},
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
    ss.setdefault("q_id", 0)
    ss.setdefault("level_choice", 1)  # selectbox key-controlled


init_state()


# ---------- Metrics helpers ----------
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


# ---------- Options / distractors ----------
def make_options(correct: int, kind: str = "generic", a=None, b=None):
    opts = {correct}

    for d in (-1, 1, -2, 2, -10, 10):
        opts.add(correct + d)

    s = str(abs(correct))
    if len(s) >= 2:
        rev = int(s[::-1])
        opts.add(rev if correct >= 0 else -rev)

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

    opts = list(opts)
    opts.remove(correct)
    random.shuffle(opts)
    selected = [correct] + opts[: N_OPTIONS - 1]
    random.shuffle(selected)
    return selected


def avoid_repeat(prompt_key: str) -> bool:
    if prompt_key == st.session_state.last_prompt:
        return False
    st.session_state.last_prompt = prompt_key
    return True


# ---------- Render helpers ----------
def render_vertical_expression(nums, ops):
    """
    Align EVERYTHING including the first operand by giving it a blank operator slot.
    """
    width = max(len(str(n)) for n in nums)
    lines = [f"  {str(nums[0]).rjust(width)}"]
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
            display = f"{a} + {b}"
            correct = a + b
            kind = "addsub"
            prompt_key = display

        elif chosen == "sub":
            a, b = random.randint(2, 99), random.randint(2, 99)
            if b > a:
                a, b = b, a
            display = f"{a} − {b}"
            correct = a - b
            kind = "addsub"
            prompt_key = display

        elif chosen == "mul":
            a, b = random.randint(2, 12), random.randint(2, 12)
            display = f"{a} × {b}"
            correct = a * b
            kind = "mul"
            prompt_key = display

        elif chosen == "div_exact":
            b = random.randint(2, 12)
            q = random.randint(2, 12)
            a = b * q
            display = f"{a} ÷ {b}"
            correct = q
            kind = "div_quot"
            prompt_key = display

        elif chosen == "div_quot":
            b = random.randint(2, 12)
            q = random.randint(2, 20)
            r = random.randint(1, b - 1)
            a = b * q + r
            display = f"{a} ÷ {b} (quociente)"
            correct = a // b
            kind = "div_quot"
            prompt_key = display

        else:
            a, b = random.randint(2, 99), random.randint(2, 99)
            display = f"{a} + {b}"
            correct = a + b
            kind = "generic"
            prompt_key = display

        if avoid_repeat(prompt_key):
            break

    options = make_options(correct, kind=kind, a=a, b=b)
    return {"display": display, "correct": correct, "options": options, "tag": f"Fatos Atômicos — {mode_key}", "kind": "inline"}


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
    return {"display": display, "correct": correct, "options": options, "tag": "Dois Passos — 3 operandos", "kind": "inline"}


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

        else:
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
    return {"display": display, "correct": correct, "options": options, "tag": f"Multioperação — {mode_key}", "kind": "vertical"}


def generate_problem(level: int, atomic_mode: str, multi_mode: str, n_operands: int):
    if level == 1:
        return gen_atomic(atomic_mode)
    if level == 2:
        return gen_two_steps()
    return gen_multiop(multi_mode, n_operands)


# ---------- Contextual reference + guidance ----------
def _targets_text_atomic(atomic_mode: str):
    if atomic_mode in ["Soma", "Subtração"]:
        return "Iniciante: 2,0–3,0 s | Intermediário: 1,2–2,0 s | Fluência alta: < 1,0 s"
    if atomic_mode == "Multiplicação":
        return "Iniciante: 2,5–4,0 s | Intermediário: 1,5–2,5 s | Fluência alta: < 1,2 s"
    if atomic_mode == "Divisão exata":
        return "Iniciante: 3,0–4,5 s | Intermediário: 2,0–3,0 s | Fluência alta: < 1,8 s"
    if atomic_mode == "Divisão com resto (só quociente)":
        return "Iniciante: 3,5–5,0 s | Intermediário: 2,5–3,5 s | Fluência alta: < 2,0 s"
    return "Modo misto: use as metas dos submodos como referência."


def recomendar_progresso(level, atomic_mode, n_operands):
    """
    Always return a recommendation string and a boolean 'should_advance' (best-effort).
    Uses rolling accuracy if available; otherwise uses total accuracy.
    Uses RT mediana if available.
    """
    total = len(st.session_state.history)
    if total == 0:
        return "Responda algumas questões para eu recomendar com base em dados.", False

    # Accuracy: prefer rolling if available
    roll = rolling_accuracy_percent()
    correct_total = sum(1 for r in st.session_state.history if r["erro"] == "correto")
    acc_total = 100.0 * correct_total / total
    acc_used = roll if roll is not None else acc_total
    acc_label = "acurácia rolante" if roll is not None else "acurácia total"

    rt_m, rt_med = rt_mean_median()
    if rt_med is None:
        return "Sem dados de tempo suficientes para recomendar.", False

    # Pick target thresholds
    target = None
    context = ""
    if level == 1:
        if atomic_mode == "Mista":
            # In mixed, be conservative: require decent speed and accuracy
            context = "Fatos Atômicos (Mista)"
            # Use intermediate upper bound of soma/sub as a proxy
            target = {"beginner_hi": 3.5, "intermediate_hi": 2.5, "advanced_hi": 1.5}
        else:
            context = f"Fatos Atômicos ({atomic_mode})"
            target = TARGETS_ATOMIC.get(atomic_mode)

    elif level == 2:
        context = "Dois Passos"
        target = TARGETS_TWO_STEPS

    else:
        context = f"Multioperação ({n_operands} operandos)"
        target = TARGETS_MULTIOP.get(n_operands)

    # Decision rule:
    # - Must have accuracy >= 90% to consider moving on
    # - Must have median RT <= intermediate_hi to move on
    # Otherwise: train more here
    should_advance = (acc_used >= 90.0) and (rt_med <= target["intermediate_hi"])

    if acc_used < 90.0:
        msg = (
            f"Recomendação: **treine mais aqui**.\n\n"
            f"Motivo: sua {acc_label} está em **{acc_used:.1f}%** (meta: ≥ 90%)."
        )
        return msg, False

    if rt_med > target["intermediate_hi"]:
        msg = (
            f"Recomendação: **treine mais aqui**.\n\n"
            f"Motivo: sua **RT mediana** está em **{rt_med:.2f}s**, acima da meta intermediária "
            f"(≤ {target['intermediate_hi']:.2f}s) para **{context}**."
        )
        return msg, False

    # Advance
    msg = (
        f"Recomendação: **pode seguir para o próximo nível**.\n\n"
        f"Você está com {acc_label} **{acc_used:.1f}%** e **RT mediana {rt_med:.2f}s** "
        f"(meta intermediária: ≤ {target['intermediate_hi']:.2f}s)."
    )
    return msg, True


def mostrar_referencia_e_navegacao(level, atomic_mode, multi_mode, n_operands):
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

    # Contextual targets
    st.markdown("### Metas de RT (por este modo)")
    if level == 1:
        st.write(_targets_text_atomic(atomic_mode))
        st.caption("Regra: velocidade só conta se a acurácia rolante estiver em **90% ou mais**.")
    elif level == 2:
        st.write("Iniciante: 4–6 s | Intermediário: 3–4 s | Avançado: < 3 s")
        st.caption("Regra: velocidade só conta se a acurácia rolante estiver em **90% ou mais**.")
    else:
        t = TARGETS_MULTIOP.get(n_operands)
        st.write(
            f"Iniciante: {t['beginner_hi']-3:.0f}–{t['beginner_hi']:.0f} s | "
            f"Intermediário: {t['advanced_hi']:.0f}–{t['intermediate_hi']:.1f} s | "
            f"Avançado: < {t['advanced_hi']:.0f} s"
        )
        st.caption("Regra: velocidade só conta se a acurácia rolante estiver em **90% ou mais**.")

    # Recommendation (always shown)
    rec, should_advance = recomendar_progresso(level, atomic_mode, n_operands)
    st.markdown("### Próximo passo")
    st.info(rec)

    # Navigation buttons (keep selectbox too)
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if st.button("⬅ Voltar nível anterior", use_container_width=True, disabled=(level <= 1)):
            st.session_state.level_choice = max(1, level - 1)
            st.session_state.current_problem = None
            st.session_state.q_id += 1
            st.rerun()
    with nav_right:
        if st.button("Seguir para próximo nível ➡", use_container_width=True, disabled=(level >= 3)):
            st.session_state.level_choice = min(3, level + 1)
            st.session_state.current_problem = None
            st.session_state.q_id += 1
            st.rerun()


# ---------------- UI ----------------
st.title("Rapid Number Forge — PT (botões)")

level = st.selectbox(
    "Opção do jogo",
    list(LEVELS.keys()),
    key="level_choice",
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
    st.caption(f"Janela rolante: {WINDOW}")

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
display = prob["display"]
correct = prob["correct"]
options = prob["options"]
tag = prob["tag"]
kind = prob["kind"]

st.caption(tag)

if kind == "vertical":
    st.code(display)
else:
    st.subheader(display)

# Answer buttons (one click = answer)
btn_cols = st.columns(3)
clicked_value = None
for i, opt in enumerate(options):
    with btn_cols[i]:
        if st.button(str(opt), key=f"opt_{st.session_state.q_id}_{i}", use_container_width=True):
            clicked_value = opt

if clicked_value is not None:
    rt = time.time() - st.session_state.start_time
    st.session_state.last_rt = rt

    is_correct = (clicked_value == correct)
    st.session_state.history.append(
        {
            "tag": tag,
            "conta": display if kind != "vertical" else "(multioperação)",
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Respostas", f"{total}")
c2.metric("Acurácia total", f"{acc_total:.1f}%")
c3.metric("Melhor rolante", f"{st.session_state.best_rolling:.1f}%")
c4.metric("Último tempo", "—" if st.session_state.last_rt is None else f"{st.session_state.last_rt:.2f}s")

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

# Contextual reference + always-on guidance + navigation buttons (END OF PAGE)
mostrar_referencia_e_navegacao(
    level=level,
    atomic_mode=atomic_mode,
    multi_mode=multi_mode,
    n_operands=n_operands,
        )
