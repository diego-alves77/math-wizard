import random
import time
from collections import deque, defaultdict
from statistics import mean, median
from pathlib import Path
import json
import hashlib

import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# Mago da Matemática — Streamlit
# ============================================================

WINDOW = 10
N_OPTIONS = 3
GAME_TITLE = "Mago da Matemática"

LEVELS = {1: "Conta Simples", 2: "Conta Dupla", 3: "Memória"}

# UI labels (requested rename: Mista -> Misto)
ATOMIC_MODES = [
    "Soma",
    "Subtração",
    "Multiplicação",
    "Divisão exata",
    "Divisão com resto (só quociente)",
    "Misto",
    "Misto Desafiador",
]

ATOMIC_MODE_TO_CODE = {
    "Soma": "add",
    "Subtração": "sub",
    "Multiplicação": "mul",
    "Divisão exata": "div_exact",
    "Divisão com resto (só quociente)": "div_quot",
    "Misto": "mixed",
    "Misto Desafiador": "mixed_hard",
}

MULTIOP_MODES = {
    "Soma": "sum",
    "Subtração": "sub",
    "Soma–Subtração": "sumsub",
    "Multiplicação": "mul",
}

# ------------------------------------------------------------
# Difficulty ladders (finite, human-oriented)
# Each stage is Roman numeral I, II, III...
# ------------------------------------------------------------

# Soma stages: ranges for (a, b) non-negative
SOMA_STAGES = [
    (0, 1, 0, 1),     # I
    (0, 3, 0, 1),     # II
    (0, 3, 0, 3),     # III
    (0, 7, 0, 3),     # IV
    (0, 7, 0, 7),     # V
    (0, 15, 0, 7),    # VI
    (0, 15, 0, 15),   # VII
    (0, 31, 0, 15),   # VIII
    (0, 31, 0, 31),   # IX
]

# Subtração stages: keep result mostly non-negative early, allow negatives later
SUB_STAGES = [
    ("nonneg", 0, 5),     # I
    ("nonneg", 0, 9),     # II
    ("nonneg", 0, 15),    # III
    ("nonneg", 0, 31),    # IV
    ("mixed", -31, 31),   # V
    ("mixed", -63, 63),   # VI
]

# Multiplicação stages: implement your requested progression
# tuple: (small_max, big_max, both_big)
MUL_STAGES = [
    (5, 5, True),     # I  both 1..5
    (7, 7, True),     # II
    (9, 9, True),     # III
    (12, 12, True),   # IV
    (5, 15, False),   # V  one in 1..5, other in 1..15
    (7, 15, False),   # VI
    (9, 15, False),   # VII
    (12, 15, False),  # VIII
    (15, 15, True),   # IX both in 1..15
]

# Divisão exata: slow growth, start with bigger quotients
DIV_EXACT_STAGES = [
    (2, 3, 1, 5),     # I
    (2, 4, 1, 6),     # II
    (2, 5, 1, 7),     # III
    (2, 6, 1, 8),     # IV
    (2, 8, 1, 9),     # V
    (2, 10, 1, 10),   # VI
]

# Divisão com resto (quociente): similar but allow remainder; start with larger quotients
DIV_QUOT_STAGES = [
    (2, 3, 1, 6),     # I
    (2, 4, 1, 7),     # II
    (2, 5, 1, 8),     # III
    (2, 6, 1, 9),     # IV
    (2, 8, 1, 10),    # V
    (2, 10, 1, 12),   # VI
]

# Conta Dupla: (a range) and (b,c range) grow slowly (finite)
TWO_STEPS_STAGES = [
    (5, 30, 2, 9),     # I
    (10, 50, 2, 12),   # II
    (20, 80, 3, 15),   # III
    (30, 99, 4, 20),   # IV
]

# Memória: stages increase operands and ranges
# Each stage: (n_operands, value_min, value_max)
MEMORY_STAGES = [
    (3, 2, 9),     # I
    (4, 2, 9),     # II
    (5, 2, 9),     # III
    (6, 2, 9),     # IV
    (6, 2, 12),    # V
    (6, 2, 15),    # VI
]

# ------------------------------------------------------------
# Scoring: linear by game type, exponential by Roman stage
# ------------------------------------------------------------
MODE_BASE_POINTS = {
    "Conta Simples": 10,
    "Conta Dupla": 18,
    "Memória": 25,
}
SCORE_GROWTH = 1.35

# ------------------------------------------------------------
# Adaptive profile for Misto / Misto Desafiador
# ------------------------------------------------------------
OPS_BASE = ["add", "sub", "mul", "div_exact", "div_quot"]
OPS_BASELINE = {op: 1.0 for op in OPS_BASE}
ADAPT_RECENT_N = 30
ADAPT_ALPHA = 1.6
ADAPT_MIN_BOOST = 0.85
ADAPT_MAX_BOOST = 2.25

# ------------------------------------------------------------
# Login (lightweight) - per-user persistence of score + stages
# ------------------------------------------------------------
USERS_DB = Path("users_db.json")

def _load_users():
    if USERS_DB.exists():
        try:
            return json.loads(USERS_DB.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_users(db):
    USERS_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def login_gate():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user", None)

    if st.session_state["logged_in"]:
        return

    st.title(GAME_TITLE)
    st.subheader("Entrar")

    db = _load_users()
    known = sorted(db.keys())

    choice = st.selectbox("Usuário (selecione ou crie um novo)", ["<novo usuário>"] + known)
    if choice == "<novo usuário>":
        username = st.text_input("Novo usuário")
    else:
        username = choice

    password = st.text_input("Senha", type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Entrar", type="primary"):
            if not username or not password:
                st.error("Preencha usuário e senha.")
                st.stop()

            if username in db:
                if db[username].get("pw") == _hash_pw(password):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Senha incorreta.")
                    st.stop()
            else:
                db[username] = {
                    "pw": _hash_pw(password),
                    "score": 0,
                    "stages": {},
                }
                _save_users(db)
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.rerun()
    with c2:
        st.caption("Login simples (não é segurança bancária).")

def persist_user_state():
    u = st.session_state.get("user")
    if not u:
        return
    db = _load_users()
    if u not in db:
        return
    db[u]["score"] = int(st.session_state.get("score", 0))
    db[u]["stages"] = dict(st.session_state.get("stages", {}))
    _save_users(db)

def load_user_state_into_session():
    u = st.session_state.get("user")
    if not u:
        return
    db = _load_users()
    if u not in db:
        return
    st.session_state["score"] = int(db[u].get("score", 0))
    st.session_state["stages"] = dict(db[u].get("stages", {}))

# ------------------------------------------------------------
# Roman numerals + formatting helpers
# ------------------------------------------------------------
ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

def roman(n: int) -> str:
    n = max(1, int(n))
    if n <= len(ROMANS):
        return ROMANS[n - 1]
    return str(n)

def fmt_addsub(a: int, b: int) -> str:
    # Never show "a + -b"
    if b < 0:
        return f"{a} − {abs(b)}"
    return f"{a} + {b}"

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
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

    ss.setdefault("score", 0)

    # stages stored in one dict so we can persist easily: key -> int stage
    ss.setdefault("stages", {})

    ss.setdefault("profile", {})
    prof = ss["profile"]
    if "op_recent" not in prof:
        prof["op_recent"] = {op: deque(maxlen=ADAPT_RECENT_N) for op in OPS_BASE}
    if "error_counts" not in prof:
        prof["error_counts"] = defaultdict(int)
    if "op_error_counts" not in prof:
        prof["op_error_counts"] = {op: defaultdict(int) for op in OPS_BASE}

init_state()

# ------------------------------------------------------------
# Stage keys
# ------------------------------------------------------------
def stage_key_for(level: int, mode_label: str, submode_label: str | None = None) -> str:
    if level == 1:
        # per atomic submode
        return f"L1::{submode_label}"
    if level == 2:
        return "L2::Conta Dupla"
    return f"L3::{mode_label}"  # "Memória" here

def get_stage(level: int, mode_label: str, submode_label: str | None = None) -> int:
    k = stage_key_for(level, mode_label, submode_label)
    st.session_state["stages"].setdefault(k, 1)
    return int(st.session_state["stages"][k])

def set_stage(level: int, mode_label: str, stage: int, submode_label: str | None = None):
    k = stage_key_for(level, mode_label, submode_label)
    st.session_state["stages"][k] = int(max(1, stage))

# ------------------------------------------------------------
# Metrics / analysis
# ------------------------------------------------------------
def rolling_accuracy_percent():
    rc = st.session_state["rolling_correct"]
    if len(rc) < WINDOW:
        return None
    return 100.0 * sum(rc) / WINDOW

def rt_mean_median_recent(n=WINDOW):
    hist = st.session_state["history"]
    if not hist:
        return None, None
    rts = [r["rt_s"] for r in hist[-n:]]
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

def memory_kb_for_problem(nums, ops) -> float:
    tokens = [str(n) for n in nums] + [str(o) for o in ops]
    total_bytes = sum(len(t.encode("utf-8")) for t in tokens)
    return total_bytes / 1024.0

def render_vertical_expression(nums, ops):
    width = max(len(str(n)) for n in nums)
    lines = [f"  {str(nums[0]).rjust(width)}"]
    for op, n in zip(ops, nums[1:]):
        lines.append(f"{op} {str(n).rjust(width)}")
    lines.append("-" * (width + 2))
    return "\n".join(lines)

# ------------------------------------------------------------
# Options (3 buttons)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Adaptive profile (Misto)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Targets and auto-progression
# ------------------------------------------------------------
def target_rt_for(level: int, submode: str, stage_idx: int, n_operands: int | None = None) -> float:
    """
    Higher stage => slightly higher acceptable RT (because difficulty increases).
    """
    stage_idx = max(1, int(stage_idx))
    if level == 1:
        if submode in ("Divisão exata", "Divisão com resto (só quociente)"):
            base = 2.6
        elif submode == "Multiplicação":
            base = 2.1
        elif submode in ("Misto", "Misto Desafiador"):
            base = 2.4
        else:
            base = 1.8
        return base + 0.25 * (stage_idx - 1)
    if level == 2:
        base = 3.2
        return base + 0.35 * (stage_idx - 1)
    # Memória
    base = 4.0
    if n_operands is not None:
        base += 0.35 * max(0, n_operands - 3)
    return base + 0.35 * (stage_idx - 1)

def stage_max_for(level: int, submode: str | None = None) -> int:
    if level == 1:
        if submode == "Soma":
            return len(SOMA_STAGES)
        if submode == "Subtração":
            return len(SUB_STAGES)
        if submode == "Multiplicação":
            return len(MUL_STAGES)
        if submode == "Divisão exata":
            return len(DIV_EXACT_STAGES)
        if submode == "Divisão com resto (só quociente)":
            return len(DIV_QUOT_STAGES)
        # Misto / Misto Desafiador use same cap (based on max of base ops)
        return max(len(SOMA_STAGES), len(SUB_STAGES), len(MUL_STAGES), len(DIV_EXACT_STAGES), len(DIV_QUOT_STAGES))
    if level == 2:
        return len(TWO_STEPS_STAGES)
    return len(MEMORY_STAGES)

def should_advance(level: int, submode: str, stage_idx: int, n_operands: int | None = None) -> bool:
    roll = rolling_accuracy_percent()
    if roll is None or roll < 90.0:
        return False
    _, rt_med = rt_mean_median_recent(WINDOW)
    if rt_med is None:
        return False
    return rt_med <= target_rt_for(level, submode, stage_idx, n_operands=n_operands)

# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------
def add_points(level: int, stage_idx: int, correct: bool):
    if not correct:
        return
    mode_label = LEVELS[int(level)]
    base = MODE_BASE_POINTS.get(mode_label, 10)
    mult = SCORE_GROWTH ** max(0, int(stage_idx) - 1)
    pts = int(round(base * mult))
    st.session_state["score"] = int(st.session_state.get("score", 0)) + pts

# ------------------------------------------------------------
# Generators per stage
# ------------------------------------------------------------
def gen_add(stage_idx: int, hard: bool):
    stage_idx = min(stage_idx, len(SOMA_STAGES))
    a_min, a_max, b_min, b_max = SOMA_STAGES[stage_idx - 1]
    a = random.randint(a_min, a_max)
    b = random.randint(b_min, b_max)
    if hard and random.random() < 0.25:
        # allow negative b in desafiador, but display will avoid "+ -"
        b = -b
    display = fmt_addsub(a, b)
    correct = a + b
    return display, correct, "addsub", [a, b], ["+"], "add", a, b

def gen_sub(stage_idx: int, hard: bool):
    stage_idx = min(stage_idx, len(SUB_STAGES))
    mode, lo, hi = SUB_STAGES[stage_idx - 1]
    if mode == "nonneg":
        a = random.randint(0, hi)
        b = random.randint(0, hi)
        if b > a:
            a, b = b, a
    else:
        a = random.randint(lo, hi)
        b = random.randint(lo, hi)
    display = f"{a} − {b}"
    correct = a - b
    return display, correct, "addsub", [a, b], ["−"], "sub", a, b

def gen_mul(stage_idx: int, hard: bool):
    stage_idx = min(stage_idx, len(MUL_STAGES))
    small_max, big_max, both_big = MUL_STAGES[stage_idx - 1]
    if both_big:
        a = random.randint(1, big_max)
        b = random.randint(1, big_max)
    else:
        small = random.randint(1, small_max)
        big = random.randint(1, big_max)
        if random.random() < 0.5:
            a, b = small, big
        else:
            a, b = big, small
    if hard and random.random() < 0.20:
        a = -a
    if hard and random.random() < 0.20:
        b = -b
    display = f"{a} × {b}"
    correct = a * b
    return display, correct, "mul", [a, b], ["×"], "mul", a, b

def gen_div_exact(stage_idx: int, hard: bool):
    stage_idx = min(stage_idx, len(DIV_EXACT_STAGES))
    dmin, dmax, qmin, qmax = DIV_EXACT_STAGES[stage_idx - 1]
    b = random.randint(dmin, dmax)
    q = random.randint(qmin, qmax)
    a = b * q
    if hard and random.random() < 0.15:
        a = -a
    display = f"{a} ÷ {b}"
    correct = int(a / b)
    return display, correct, "div_quot", [a, b], ["÷"], "div_exact", a, b

def gen_div_quot(stage_idx: int, hard: bool):
    stage_idx = min(stage_idx, len(DIV_QUOT_STAGES))
    dmin, dmax, qmin, qmax = DIV_QUOT_STAGES[stage_idx - 1]
    b = random.randint(dmin, dmax)
    q = random.randint(qmin, qmax)
    r = random.randint(1, b - 1)
    a = b * q + r
    if hard and random.random() < 0.15:
        a = -a
    display = f"{a} ÷ {b}"
    correct = a // b
    return display, correct, "div_quot", [a, b], ["÷"], "div_quot", a, b

def gen_atomic(submode_label: str, stage_idx: int):
    code = ATOMIC_MODE_TO_CODE[submode_label]
    while True:
        if code in ("mixed", "mixed_hard"):
            hard = (code == "mixed_hard")
            # adaptive selection of base op
            op = weighted_choice(adaptive_op_weights())
            # stage for mixed uses current stage_idx (clamped per op)
            if op == "add":
                display, correct, kind, nums, ops, op_code, a, b = gen_add(stage_idx, hard)
            elif op == "sub":
                display, correct, kind, nums, ops, op_code, a, b = gen_sub(stage_idx, hard)
            elif op == "mul":
                display, correct, kind, nums, ops, op_code, a, b = gen_mul(stage_idx, hard)
            elif op == "div_exact":
                display, correct, kind, nums, ops, op_code, a, b = gen_div_exact(stage_idx, hard)
            else:
                display, correct, kind, nums, ops, op_code, a, b = gen_div_quot(stage_idx, hard)
            prompt_key = f"{display}|{op}|{stage_idx}|{'H' if hard else 'N'}"
        else:
            hard = False
            if code == "add":
                display, correct, kind, nums, ops, op_code, a, b = gen_add(stage_idx, hard)
            elif code == "sub":
                display, correct, kind, nums, ops, op_code, a, b = gen_sub(stage_idx, hard)
            elif code == "mul":
                display, correct, kind, nums, ops, op_code, a, b = gen_mul(stage_idx, hard)
            elif code == "div_exact":
                display, correct, kind, nums, ops, op_code, a, b = gen_div_exact(stage_idx, hard)
            else:
                display, correct, kind, nums, ops, op_code, a, b = gen_div_quot(stage_idx, hard)
            prompt_key = f"{display}|{code}|{stage_idx}"

        if avoid_repeat(prompt_key):
            break

    options = make_options(correct, kind=kind, a=a, b=b)
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": f"Conta Simples — {submode_label} — {roman(stage_idx)}",
        "kind": "inline",
        "nums": nums,
        "ops": ops,
        "op_code": op_code,
        "submode": submode_label,
        "stage": stage_idx,
    }

def gen_two_steps(stage_idx: int):
    stage_idx = min(stage_idx, len(TWO_STEPS_STAGES))
    a_lo, a_hi, bc_lo, bc_hi = TWO_STEPS_STAGES[stage_idx - 1]
    while True:
        a = random.randint(a_lo, a_hi)
        b = random.randint(bc_lo, bc_hi)
        c = random.randint(bc_lo, bc_hi)
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

        if avoid_repeat(display + f"|{stage_idx}"):
            break

    options = make_options(correct, kind="addsub")
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": f"Conta Dupla — {roman(stage_idx)}",
        "kind": "inline",
        "nums": [a, b, c],
        "ops": [op1, op2],
        "op_code": "two_steps",
        "submode": "Conta Dupla",
        "stage": stage_idx,
    }

def gen_memory(mode_key: str, stage_idx: int):
    stage_idx = min(stage_idx, len(MEMORY_STAGES))
    n_operands, vmin, vmax = MEMORY_STAGES[stage_idx - 1]
    mode = MULTIOP_MODES[mode_key]
    while True:
        nums = [random.randint(vmin, vmax) for _ in range(n_operands)]

        if mode == "sum":
            ops = ["+"] * (n_operands - 1)
            correct = sum(nums)
            kind = "addsub"
        elif mode == "sub":
            nums[0] = random.randint(max(vmin, 10), max(vmax, 30))
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
            # Multiplication in memory: keep numbers smaller
            nums = [random.randint(2, min(9, vmax)) for _ in range(n_operands)]
            ops = ["×"] * (n_operands - 1)
            correct = 1
            for x in nums:
                correct *= x
            kind = "mul"

        prompt_key = f"MEM:{mode_key}:{stage_idx}:{','.join(map(str, nums))}:{''.join(ops)}"
        if avoid_repeat(prompt_key):
            break

    display = render_vertical_expression(nums, ops)
    options = make_options(correct, kind=kind)
    return {
        "display": display,
        "correct": correct,
        "options": options,
        "tag": f"Memória — {mode_key} — {roman(stage_idx)}",
        "kind": "vertical",
        "nums": nums,
        "ops": ops,
        "op_code": f"mem_{mode}",
        "submode": f"Memória — {mode_key}",
        "stage": stage_idx,
        "n_operands": n_operands,
    }

def generate_problem(level: int, atomic_mode: str, memory_mode: str):
    if level == 1:
        stage = get_stage(1, "Conta Simples", atomic_mode)
        stage = min(stage, stage_max_for(1, atomic_mode))
        return gen_atomic(atomic_mode, stage)
    if level == 2:
        stage = get_stage(2, "Conta Dupla", None)
        stage = min(stage, stage_max_for(2, None))
        return gen_two_steps(stage)
    stage = get_stage(3, "Memória", "Memória")
    stage = min(stage, stage_max_for(3, None))
    return gen_memory(memory_mode, stage)

# ------------------------------------------------------------
# Navigation progression (requested sequence):
# each Conta Simples submode -> Conta Dupla -> Memória
# ------------------------------------------------------------
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

def goto_prev(level: int, atomic_mode: str):
    new_level, new_atomic = _prev_stage(level, atomic_mode)
    st.session_state["level_choice"] = int(new_level)
    if int(new_level) == 1 and new_atomic is not None:
        st.session_state["atomic_mode_choice"] = new_atomic
    st.session_state["scroll_to_challenge"] = False
    st.session_state["current_problem"] = None
    st.session_state["q_id"] += 1

def goto_next(level: int, atomic_mode: str):
    new_level, new_atomic = _next_stage(level, atomic_mode)
    st.session_state["level_choice"] = int(new_level)
    if int(new_level) == 1 and new_atomic is not None:
        st.session_state["atomic_mode_choice"] = new_atomic
    st.session_state["scroll_to_challenge"] = True
    st.session_state["current_problem"] = None
    st.session_state["q_id"] += 1

# ------------------------------------------------------------
# Reference + orientation
# ------------------------------------------------------------
def show_reference(level: int, submode_label: str, stage_idx: int, mem_kb: float, n_operands: int | None):
    st.divider()
    st.subheader("Referência e Orientação")

    st.markdown(
        "**Siglas e termos (nesta página)**\n"
        "- **RT** = *Tempo de Resposta* (tempo entre aparecer a conta e você clicar na resposta).\n"
        "- **RT média** = média aritmética dos tempos (pode ser distorcida por respostas muito lentas).\n"
        "- **RT mediana** = valor central dos tempos ordenados (representa melhor seu ritmo típico).\n"
        "- **Acurácia rolante** = porcentagem de acertos nas **últimas 10** respostas."
    )

    st.markdown("### Meta de RT (neste nível)")
    target = target_rt_for(level, submode_label, stage_idx, n_operands=n_operands)
    st.write(f"Meta intermediária neste nível: **≤ {target:.2f} s** (com acurácia rolante ≥ 90%).")

    st.markdown("### Carga de Memória")
    st.write(f"Estimativa: **{mem_kb:.4f} kB**")
    st.caption("Bytes UTF-8 de **operandos + operadores**, convertidos para kB (1 kB = 1024 bytes).")

def recommend_and_autoadvance(level: int, submode_label: str, stage_idx: int, n_operands: int | None):
    st.markdown("### Próximo passo")
    max_stage = stage_max_for(level, submode_label if level == 1 else None)
    roll = rolling_accuracy_percent()
    _, rt_med = rt_mean_median_recent(WINDOW)

    if roll is None or rt_med is None:
        st.info("Responda mais questões para eu recomendar com base em dados.")
        return stage_idx

    if roll < 90.0:
        st.info(f"Recomendação: **treinar mais aqui**. Acurácia rolante: **{roll:.1f}%** (meta: ≥ 90%).")
        return stage_idx

    target = target_rt_for(level, submode_label, stage_idx, n_operands=n_operands)
    if rt_med > target:
        st.info(
            f"Recomendação: **treinar mais aqui**. RT mediana (últimas {WINDOW}): **{rt_med:.2f}s** "
            f"(meta: ≤ {target:.2f}s)."
        )
        return stage_idx

    if stage_idx < max_stage:
        st.success("Você dominou este nível com tranquilidade. **Avançando automaticamente para o próximo (Roman).**")
        new_stage = stage_idx + 1
        if level == 1:
            set_stage(1, "Conta Simples", new_stage, submode_label)
        elif level == 2:
            set_stage(2, "Conta Dupla", new_stage, None)
        else:
            set_stage(3, "Memória", new_stage, "Memória")
        persist_user_state()
        return new_stage

    st.success("Você já está no último nível desta trilha. Continue refinando para ganhar consistência.")
    return stage_idx

# ============================================================
# APP START
# ============================================================
login_gate()
if st.session_state.get("logged_in"):
    # Load user state once
    if not st.session_state.get("_loaded_user_state", False):
        load_user_state_into_session()
        st.session_state["_loaded_user_state"] = True

st.title(GAME_TITLE)

st.metric("Pontuação", str(st.session_state.get("score", 0)))

st.caption(
    "Treine cálculo mental com respostas por clique. Você avança automaticamente dentro de cada tipo de jogo "
    "quando mantém **acurácia rolante ≥ 90%** e **RT mediana** dentro da meta do nível."
)

# ------------------------------------------------------------
# Menu
# ------------------------------------------------------------
level = st.selectbox(
    "Opção do jogo",
    list(LEVELS.keys()),
    key="level_choice",
    format_func=lambda x: f"{x} — {LEVELS[x]}",
)

atomic_mode = st.session_state["atomic_mode_choice"]
memory_mode = st.session_state["multi_mode_choice"]

if int(level) == 1:
    atomic_mode = st.selectbox(
        "Modo (Conta Simples)",
        ATOMIC_MODES,
        key="atomic_mode_choice",
    )
elif int(level) == 3:
    memory_mode = st.selectbox(
        "Tipo (Memória)",
        list(MULTIOP_MODES.keys()),
        key="multi_mode_choice",
    )

st.divider()

# show current stage for each game type (requested)
cs_stage = get_stage(1, "Conta Simples", atomic_mode)
cd_stage = get_stage(2, "Conta Dupla", None)
mem_stage = get_stage(3, "Memória", "Memória")

s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Conta Simples (nível)", roman(min(cs_stage, stage_max_for(1, atomic_mode))))
with s2:
    st.metric("Conta Dupla (nível)", roman(min(cd_stage, stage_max_for(2, None))))
with s3:
    st.metric("Memória (nível)", roman(min(mem_stage, stage_max_for(3, None))))

# ------------------------------------------------------------
# Controls
# ------------------------------------------------------------
colA, colB = st.columns([1, 1])

with colA:
    if not st.session_state["started"]:
        if st.button("▶️ Iniciar", type="primary"):
            st.session_state["started"] = True
            st.session_state["current_problem"] = None
            st.session_state["q_id"] += 1
            st.rerun()
    else:
        if st.button("⏹️ Reiniciar", type="secondary"):
            # Keep login + selections; reset gameplay
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
                "scroll_to_challenge",
                "profile",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            init_state()
            load_user_state_into_session()
            st.rerun()

with colB:
    st.caption(f"Janela rolante: {WINDOW}")

if not st.session_state["started"]:
    st.stop()

# ------------------------------------------------------------
# Generate problem
# ------------------------------------------------------------
if st.session_state["current_problem"] is None:
    st.session_state["current_problem"] = generate_problem(int(level), atomic_mode, memory_mode)
    st.session_state["start_time"] = time.time()
    st.session_state["q_id"] += 1

prob = st.session_state["current_problem"]
display = prob["display"]
correct = prob["correct"]
options = prob["options"]
tag = prob["tag"]
kind = prob["kind"]
nums = prob["nums"]
ops = prob["ops"]
op_code = prob.get("op_code", "unknown")
submode_label = prob.get("submode", "")
stage_idx = int(prob.get("stage", 1))
n_operands = prob.get("n_operands", None)

mem_kb = memory_kb_for_problem(nums, ops)

# ------------------------------------------------------------
# Scroll anchor (Avançar)
# ------------------------------------------------------------
st.markdown('<div id="challenge"></div>', unsafe_allow_html=True)
if st.session_state["scroll_to_challenge"]:
    components.html(
        "<script>"
        "const el = window.parent.document.getElementById('challenge');"
        "if (el) { el.scrollIntoView({behavior:'smooth', block:'center'}); }"
        "</script>",
        height=0,
    )
    st.session_state["scroll_to_challenge"] = False

st.caption(tag)
if kind == "vertical":
    st.code(display)
else:
    st.subheader(display)

# ------------------------------------------------------------
# Answer buttons
# ------------------------------------------------------------
btn_cols = st.columns(3)
clicked_value = None
for i, opt in enumerate(options):
    with btn_cols[i]:
        if st.button(str(opt), key=f"opt_{st.session_state['q_id']}_{i}", use_container_width=True):
            clicked_value = opt

# ------------------------------------------------------------
# Handle answer
# ------------------------------------------------------------
if clicked_value is not None:
    rt = time.time() - st.session_state["start_time"]
    st.session_state["last_rt"] = rt

    is_correct = clicked_value == correct
    err = detect_error(clicked_value, correct)
    profile_record(op_code, is_correct, err)

    # scoring
    add_points(int(level), stage_idx, is_correct)

    st.session_state["history"].append(
        {
            "tag": tag,
            "entrada": clicked_value,
            "correto": correct,
            "rt_s": rt,
            "erro": err,
            "op": op_code,
        }
    )

    st.session_state["rolling_correct"].append(is_correct)
    if len(st.session_state["rolling_correct"]) == WINDOW:
        pct = 100.0 * sum(st.session_state["rolling_correct"]) / WINDOW
        st.session_state["rolling_scores"].append(pct)
        st.session_state["best_rolling"] = max(st.session_state["best_rolling"], pct)

    # auto-advance stage if mastered (within this selected game type)
    if int(level) == 1:
        current_stage = get_stage(1, "Conta Simples", atomic_mode)
        if should_advance(1, atomic_mode, current_stage, n_operands=None):
            new_stage = min(current_stage + 1, stage_max_for(1, atomic_mode))
            set_stage(1, "Conta Simples", new_stage, atomic_mode)
    elif int(level) == 2:
        current_stage = get_stage(2, "Conta Dupla", None)
        if should_advance(2, "Conta Dupla", current_stage, n_operands=None):
            new_stage = min(current_stage + 1, stage_max_for(2, None))
            set_stage(2, "Conta Dupla", new_stage, None)
    else:
        current_stage = get_stage(3, "Memória", "Memória")
        if should_advance(3, "Memória", current_stage, n_operands=n_operands):
            new_stage = min(current_stage + 1, stage_max_for(3, None))
            set_stage(3, "Memória", new_stage, "Memória")

    persist_user_state()

    st.session_state["current_problem"] = None
    st.rerun()

# ------------------------------------------------------------
# Stats
# ------------------------------------------------------------
st.divider()

hist = st.session_state["history"]
total = len(hist)
correct_total = sum(1 for r in hist if r["erro"] == "correto")
acc_total = (100.0 * correct_total / total) if total else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Respostas", f"{total}")
m2.metric("Acurácia total", f"{acc_total:.1f}%")
m3.metric("Melhor rolante", f"{st.session_state['best_rolling']:.1f}%")
m4.metric("Último tempo", "—" if st.session_state["last_rt"] is None else f"{st.session_state['last_rt']:.2f}s")

if total:
    rts = [r["rt_s"] for r in hist]
    st.caption(f"RT média: {mean(rts):.2f}s | RT mediana: {median(rts):.2f}s")

if len(st.session_state["rolling_correct"]) < WINDOW:
    st.info(f"Acurácia rolante aparece após {WINDOW} respostas.")
else:
    current_pct = st.session_state["rolling_scores"][-1]
    st.metric(f"Acurácia rolante (últimas {WINDOW})", f"{current_pct:.1f}%")
    st.line_chart(st.session_state["rolling_scores"])

if total:
    st.subheader("Histórico recente")
    st.dataframe(hist[-20:], use_container_width=True)

# ------------------------------------------------------------
# Reference + recommendation + navigation
# ------------------------------------------------------------
show_reference(int(level), submode_label if int(level) != 2 else "Conta Dupla", stage_idx, mem_kb, n_operands)
stage_idx = recommend_and_autoadvance(int(level), submode_label if int(level) != 2 else "Conta Dupla", stage_idx, n_operands)

nav_left, nav_right = st.columns(2)
with nav_left:
    st.button(
        "⬅ Voltar",
        use_container_width=True,
        disabled=(int(level) == 1 and atomic_mode == ATOMIC_MODES[0]),
        on_click=goto_prev,
        args=(int(level), atomic_mode),
        key=f"nav_prev_{st.session_state['q_id']}",
    )
with nav_right:
    st.button(
        "Avançar ➡",
        use_container_width=True,
        disabled=(int(level) == 3),
        on_click=goto_next,
        args=(int(level), atomic_mode),
        key=f"nav_next_{st.session_state['q_id']}",
    )
