import random
import time
from collections import deque, defaultdict
from statistics import mean, median

import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
WINDOW = 10
N_OPTIONS = 3
GAME_TITLE = "Mago da Matemática"

LEVELS = {1: "Conta Simples", 2: "Conta Dupla", 3: "Memória"}

ATOMIC_MODES = [
    "Soma",
    "Subtração",
    "Multiplicação",
    "Divisão exata",
    "Divisão com resto (só quociente)",
    "Mista",
    "Misto Desafiador",
]

ATOMIC_MODE_TO_CODE = {
    "Soma": "add",
    "Subtração": "sub",
    "Multiplicação": "mul",
    "Divisão exata": "div_exact",
    "Divisão com resto (só quociente)": "div_quot",
    "Mista": "mixed",
    "Misto Desafiador": "mixed_hard",
}

MULTIOP_MODES = {
    "Soma": "sum",
    "Subtração": "sub",
    "Soma–Subtração": "sumsub",
    "Multiplicação": "mul",
}

TARGETS_ATOMIC_INTERMEDIATE_HI = {
    "Soma": 2.0,
    "Subtração": 2.0,
    "Multiplicação": 2.5,
    "Divisão exata": 3.0,
    "Divisão com resto (só quociente)": 3.5,
    "Mista": 2.5,
    "Misto Desafiador": 3.2,
}
TARGET_TWO_STEPS_INTERMEDIATE_HI = 4.0
TARGETS_MULTIOP_INTERMEDIATE_HI = {3: 4.5, 4: 6.0, 5: 8.0, 6: 10.0}

OPS_BASE = ["add", "sub", "mul", "div_exact", "div_quot"]
OPS_BASELINE = {op: 1.0 for op in OPS_BASE}

ADAPT_RECENT_N = 30
ADAPT_ALPHA = 1.6
ADAPT_MIN_BOOST = 0.85
ADAPT_MAX_BOOST = 2.25

GAME_DESC = (
    "Treine cálculo mental com respostas por clique. O jogo mede sua **acurácia** e seu **RT** "
    "(tempo de resposta) e recomenda se você deve **treinar mais** ou **avançar**. "
    "Nos modos **Mista** e **Misto Desafiador**, ele aprende seus padrões e aumenta a frequência "
    "dos tipos de conta em que você mais erra — voltando ao normal quando você melhora."
)


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

    ss.setdefault("scroll_to_challenge", False)

    ss.setdefault("level_choice", 1)
    ss.setdefault("atomic_mode_choice", ATOMIC_MODES[0])
    ss.setdefault("multi_mode_choice", "Soma")
    ss.setdefault("n_operands_choice", 3)

    ss.setdefault("profile", {})
    prof = ss["profile"]
    if "op_recent" not in prof:
        prof["op_recent"] = {op: deque(maxlen=ADAPT_RECENT_N) for op in OPS_BASE}
    if "error_counts" not in prof:
        prof["error_counts"] = defaultdict(int)
    if "op_error_counts" not in prof:
        prof["op_error_counts"] = {op: defaultdict(int) for op in OPS_BASE}


init_state()


# =========================
# NAVIGATION
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
        st.session_state["scroll_to_challenge"] = True
    else:
        new_level, new_atomic = _prev_stage(level, atomic_mode)
        st.session_state["scroll_to_challenge"] = False

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
# METRICS / HELPERS
# =========================
def rolling_accuracy_percent():
    rc = st.session_state["rolling_correct"]
    if len(rc) < WINDOW:
        return None
    return 100.0 * sum(rc) / WINDOW


def rt_mean_median():
    hist = st.session_state["history"]
    if not hist:
        return None, None
    rts = [r["rt_s"] for r in hist]
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


def avoid_repeat(prompt_key: str) -> bool:
    if prompt_key == st.session_state["last_prompt_key"]:
        return False
    st.session_state["last_prompt_key"] = prompt_key
    return True


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
    if correct in opts:
        opts.remove(correct)
    random.shuffle(opts)
    selected = [correct] + opts[: N_OPTIONS - 1]
    random.shuffle(selected)
    return selected


def render_vertical_expression(nums, ops):
    width = max(len(str(n)) for n in nums)
    lines = [f"  {str(nums[0]).rjust(width)}"]
    for op, n in zip(ops, nums[1:]):
        lines.append(f"{op} {str(n).rjust(width)}")
    lines.append("-" * (width + 2))
    return "\n".join(lines)


def memory_kb_for_problem(nums, ops) -> float:
    tokens = [str(n) for n in nums] + [str(o) for o in ops]
    total_bytes = sum(len(t.encode("utf-8")) for t in tokens)
    return total_bytes / 1024.0


# =========================
# ADAPTIVE PROFILE
# =========================
def profile_record(op_code: str, is_correct: bool, error_label: str):
    prof = st.session_state["profile"]
    prof["error_counts"][error_label] += 1
    if op_code in OPS_BASE:
        prof["op_recent"][op_code].append(1 if is_correct else 0)
        prof["op_error_counts"][op_code][error_label] += 1


def adaptive_op_weights():
    prof = st.session_state["profile"]
    weights = {}
    for op in OPS_BASE:
        recent = prof["op_recent"][op]
        acc = 1.0 if len(recent) == 0 else (sum(recent) / len(recent))
        difficulty = max(0.0, 1.0 - acc)
        mult = 1.0 + ADAPT_ALPHA * difficulty
        mult = max(ADAPT_MIN_BOOST, min(ADAPT_MAX_BOOST, mult))
        weights[op] = OPS_BASELINE[op] * mult

    total = sum(weights.values())
    if total <= 0:
        return {op: 1.0 / len(OPS_BASE) for op in OPS_BASE}
    return {op: weights[op] / total for op in OPS_BASE}


def weighted_choice(weights):
    r = random.random()
    s = 0.0
    for k, w in weights.items():
        s += w
        if r <= s:
            return k
    return next(iter(weights.keys()))


# =========================
# GENERATORS
# =========================
def gen_op_problem(op_code: str, hard: bool):
    if op_code == "add":
        a = random.randint(-199, 199) if hard else random.randint(2, 99)
        b = random.randint(-199, 199) if hard else random.randint(2, 99)
        display = f"{a} + {b}"
        correct = a + b
        kind = "addsub"
        nums, ops = [a, b], ["+"]

    elif op_code == "sub":
        if hard:
            a = random.randint(-199, 199)
            b = random.randint(-199, 199)
        else:
            a = random.randint(2, 99)
            b = random.randint(2, 99)
            if b > a:
                a, b = b, a
        display = f"{a} − {b}"
        correct = a - b
        kind = "addsub"
        nums, ops = [a, b], ["−"]

    elif op_code == "mul":
        if hard:
            a = random.randint(7, 25)
            b = random.randint(7, 25)
            if random.random() < 0.25:
                a = -a
            if random.random() < 0.25:
                b = -b
        else:
            a = random.randint(2, 12)
            b = random.randint(2, 12)
        display = f"{a} × {b}"
        correct = a * b
        kind = "mul"
        nums, ops = [a, b], ["×"]

    elif op_code == "div_exact":
        b = random.randint(2, 25) if hard else random.randint(2, 12)
        q = random.randint(2, 25) if hard else random.randint(2, 12)
        a = b * q
        if hard and random.random() < 0.20:
            a = -a
        display = f"{a} ÷ {b}"
        correct = int(a / b)
        kind = "div_quot"
        nums, ops = [a, b], ["÷"]

    else:  # div_quot
        b = random.randint(2, 25) if hard else random.randint(2, 12)
        q = random.randint(2, 40) if hard else random.randint(2, 20)
        r = random.randint(1, b - 1)
        a = b * q + r
        if hard and random.random() < 0.20:
            a = -a
        display = f"{a} ÷ {b}"
        correct = a // b
        kind = "div_quot"
        nums, ops = [a, b], ["÷"]

    prompt_key = f"{display}|{op_code}|{'H' if hard else 'N'}"
    return display, correct, kind, nums, ops, prompt_key, op_code, a, b


def gen_atomic(mode_key: str):
    code = ATOMIC_MODE_TO_CODE[mode_key]
    while True:
        if code in ("mixed", "mixed_hard"):
            hard = (code == "mixed_hard")
            op = weighted_choice(adaptive_op_weights())
            display, correct, kind, nums, ops, prompt_key, op_used, a, b = gen_op_problem(op, hard)
        else:
            display, correct, kind, nums, ops, prompt_key, op_used, a, b = gen_op_problem(code, False)
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
        "op_code": op_used,
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
        "op_code": "two_steps",
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
                correct = (correct + x) if (op == "+") else (correct - x)
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
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": f"Memória — {mode_key}",
        "kind": "vertical",
        "nums": nums,
        "ops": ops,
        "op_code": f"mem_{mode}",
    }


def generate_problem(level: int, atomic_mode: str, multi_mode: str, n_operands: int):
    if level == 1:
        return gen_atomic(atomic_mode)
    if level == 2:
        return gen_two_steps()
    return gen_multiop(multi_mode, n_operands)


# =========================
# GUIDANCE
# =========================
def targets_text_for_context(level, atomic_mode=None, n_operands=None):
    if level == 1:
        if atomic_mode in ("Soma", "Subtração"):
            return "Meta de RT: Iniciante 2,0–3,0 s | Intermediário 1,2–2,0 s | Fluência alta < 1,0 s"
        if atomic_mode == "Multiplicação":
            return "Meta de RT: Iniciante 2,5–4,0 s | Intermediário 1,5–2,5 s | Fluência alta < 1,2 s"
        if atomic_mode == "Divisão exata":
            return "Meta de RT: Iniciante 3,0–4,5 s | Intermediário 2,0–3,0 s | Fluência alta < 1,8 s"
        if atomic_mode == "Divisão com resto (só quociente)":
            return "Meta de RT: Iniciante 3,5–5,0 s | Intermediário 2,5–3,5 s | Fluência alta < 2,0 s"
        if atomic_mode == "Misto Desafiador":
            return "Meta de RT: Iniciante 3,0–5,0 s | Intermediário 2,2–3,2 s | Fluência alta < 2,0 s"
        return "Modo misto: use as metas dos submodos como referência."
    if level == 2:
        return "Meta de RT: Iniciante 4–6 s | Intermediário 3–4 s | Avançado < 3 s"
    n = int(n_operands)
    if n == 3:
        return "Meta de RT: Iniciante 4–7 s | Intermediário 3–4,5 s | Avançado < 3 s"
    if n == 4:
        return "Meta de RT: Iniciante 6–9 s | Intermediário 4–6 s | Avançado < 4 s"
    if n == 5:
        return "Meta de RT: Iniciante 8–12 s | Intermediário 6–8 s | Avançado < 6 s"
    return "Meta de RT: Iniciante 10–15 s | Intermediário 8–10 s | Avançado < 7 s"


def recommend_progress(level, atomic_mode, n_operands):
    hist = st.session_state["history"]
    if not hist:
        return "Responda algumas questões para eu recomendar com base em dados.", False

    roll = rolling_accuracy_percent()
    correct_total = sum(1 for r in hist if r["erro"] == "correto")
    acc_total = 100.0 * correct_total / len(hist)
    acc_used = roll if roll is not None else acc_total
    acc_label = "acurácia rolante" if roll is not None else "acurácia total"

    _, rt_med = rt_mean_median()
    if rt_med is None:
        return "Sem dados suficientes para recomendar.", False

    if level == 1:
        thr = TARGETS_ATOMIC_INTERMEDIATE_HI.get(atomic_mode, 2.5)
        ctx = f"Conta Simples ({atomic_mode})"
        nxt_level, nxt_atomic = _next_stage(1, atomic_mode)
        next_label = (f"{nxt_atomic} (Conta Simples)" if nxt_level == 1 else "Conta Dupla")
    elif level == 2:
        thr = TARGET_TWO_STEPS_INTERMEDIATE_HI
        ctx = "Conta Dupla"
        next_label = "Memória"
    else:
        thr = TARGETS_MULTIOP_INTERMEDIATE_HI.get(int(n_operands), 6.0)
        ctx = f"Memória ({n_operands})"
        next_label = None

    if acc_used < 90.0:
        return f"Recomendação: **treine mais aqui**.\n\nMotivo: sua {acc_label} está em **{acc_used:.1f}%** (meta: ≥ 90%).", False
    if rt_med > thr:
        return (
            f"Recomendação: **treine mais aqui**.\n\n"
            f"Motivo: sua **RT mediana** está em **{rt_med:.2f}s**, acima da meta intermediária (≤ {thr:.2f}s) "
            f"para **{ctx}**."
        ), False
    if next_label is None:
        return (
            f"Recomendação: **continue refinando**.\n\n"
            f"Você está com {acc_label} **{acc_used:.1f}%** e **RT mediana {rt_med:.2f}s** (meta: ≤ {thr:.2f}s)."
        ), True
    return (
        f"Recomendação: **pode avançar**.\n\n"
        f"Você está com {acc_label} **{acc_used:.1f}%** e **RT mediana {rt_med:.2f}s** (meta: ≤ {thr:.2f}s). "
        f"Agora siga para **{next_label}**."
    ), True


def show_reference_and_navigation(level, atomic_mode_for_path, n_operands, current_mem_kb):
    st.divider()
    st.subheader("Referência e Orientação")

    st.markdown(
        "**Siglas e termos (nesta página)**\n"
        "- **RT** = *Tempo de Resposta* (tempo entre aparecer a conta e você clicar na resposta).\n"
        "- **RT média** = média aritmética dos tempos (pode ser distorcida por respostas muito lentas).\n"
        "- **RT mediana** = valor central dos tempos ordenados (representa melhor seu ritmo típico).\n"
        "- **Acurácia rolante** = porcentagem de acertos nas **últimas 10** respostas."
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
    st.caption("Estimativa: bytes UTF-8 dos **operandos + operadores**, convertidos para kB (1 kB = 1024 bytes).")

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
            key=f"nav_prev_{st.session_state['q_id']}",
        )
    with nav_right:
        st.button(
            "Avançar ➡",
            use_container_width=True,
            disabled=(level == 3),
            on_click=goto_next,
            args=(level, atomic_mode_for_path),
            key=f"nav_next_{st.session_state['q_id']}",
        )


# =========================
# UI
# =========================
st.title(GAME_TITLE)

level = st.selectbox("Opção do jogo", list(LEVELS.keys()), key="level_choice", format_func=lambda x: f"{x} — {LEVELS[x]}")
st.caption(GAME_DESC)

atomic_mode_for_path = st.session_state["atomic_mode_choice"]
multi_mode = st.session_state["multi_mode_choice"]
n_operands = st.session_state["n_operands_choice"]

if int(level) == 1:
    atomic_mode_for_path = st.selectbox("Modo (Conta Simples)", ATOMIC_MODES, key="atomic_mode_choice")
elif int(level) == 3:
    c1, c2 = st.columns([2, 1])
    with c1:
        multi_mode = st.selectbox("Tipo (Memória)", list(MULTIOP_MODES.keys()), key="multi_mode_choice")
    with c2:
        n_operands = st.selectbox("Operandos", [3, 4, 5, 6], key="n_operands_choice")

st.divider()

colA, colB = st.columns([1, 1])
with colA:
    if not st.session_state["started"]:
     
