import time
import random
import math
import csv
import io
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import streamlit as st

# Language: pt (Portuguese)

T = {
    "APP_TITLE": "Rapid Number Forge (RNF) — Treino de Aritmética Mental",
    "TAGLINE": "Prática deliberada cronometrada + dificuldade adaptativa + feedback imediato. Ideal: 8–12 min/dia.",
    "CHOOSE_LEVEL": "Escolha um nível",
    "PICK_TODAY": "Escolha onde você quer treinar hoje. Você pode mudar de nível depois.",
    "START_LEVEL": "Iniciar Nível",
    "DEFAULT_TIME": "Tempo padrão:",
    "YOUR_SESSION_SCORE": "Sua pontuação da sessão no N{lvl}:",
    "TIP_LANDING": "Dica: muita gente vai melhor com um aquecimento curto no Nível 1 e o Nível 3 como bloco principal.",
    "SETTINGS": "Configurações",
    "LEVEL": "Nível",
    "AUTO_NEXT": "Gerar próxima questão automaticamente após enviar",
    "TIME_LIMIT": "Limite de tempo (segundos)",
    "L3_STREAM": "Comprimento do fluxo (Nível 3)",
    "L4_STEPS": "Passos (Nível 4)",
    "CALIBRATION": "Calibração",
    "CALIBRATION_CAP": "A dificuldade se ajusta automaticamente. Você também pode ajustar manualmente.",
    "DIFFICULTY": "Dificuldade",
    "LEVEL_TOOLS": "Ferramentas do nível",
    "RESET_LEVEL": "Zerar pontuação deste nível",
    "RESET_DONE": "Zerado.",
    "BACK_CHOOSER": "Voltar ao seletor de níveis",
    "RESET_SESSION": "Zerar sessão (apaga TODAS as tentativas)",
    "PROBLEM": "Problema",
    "INFO_BAR": "Limite: **{tl:.1f}s** • Dificuldade: **{diff:.1f}** • Pontuação do nível: **{score:.2f}**",
    "INPUT_MODE": "Modo de entrada",
    "ONE_TAP": "Um toque (alternativas)",
    "TYPING": "Digitação (teclado)",
    "CHOICES_PER": "Alternativas por questão",
    "ANSWER": "Resposta",
    "SUBMIT": "Enviar",
    "ENTER_INT": "Digite um número inteiro. A resposta correta era **{correct}**.",
    "TIMEOUT": "⏱️ Tempo! Você levou **{rt}**. Resposta correta: **{correct}**.",
    "CORRECT": "✅ Correto em **{rt}**. +{score:.2f} pontos. Dificuldade → {new_d:.1f}",
    "INCORRECT": "❌ Incorreto. Você respondeu **{answer}** em **{rt}**. Correto: **{correct}**. Dificuldade → {new_d:.1f}",
    "SESSION_SUMMARY": "Resumo da sessão",
    "PROBLEMS_ALL": "Problemas (total)",
    "TOTAL_SCORE_ALL": "Pontuação total (sessão)",
    "SCORE_LEVEL": "Pontuação (este nível)",
    "HISTORY": "Histórico (mais recente primeiro)",
    "DOWNLOAD_CSV": "Baixar CSV da sessão",
    "LEADERBOARD_TITLE": "🏆 Placar global — por nível",
    "LEADERBOARD_LEVEL": "Nível do placar",
    "NO_SCORES": "Ainda não há pontuações neste nível. Seja o primeiro recordista.",
    "NEW_TOP": "🔥 Novo #1 do Nível {lvl}! Você pode registrar seu apelido.",
    "ALIAS": "Seu apelido (máx. 24 caracteres)",
    "SUBMIT_LEVEL": "Enviar pontuação do Nível {lvl}",
    "SUBMITTED": "✅ Enviado! Recarregando placar…",
    "SUBMIT_FAIL": "Não foi possível enviar: {e}",
    "CONSISTENCY_TIP": "Dica: consistência ganha de maratonas. Faça 8–12 min/dia.",
}

APP_TITLE = T["APP_TITLE"]

LEVELS = {
    1: "Fatos Atômicos",
    2: "Operações em 2 Passos",
    3: "Total Corrente (Fluxo)",
    4: "Saltos Multiplicativos",
    5: "Modo Compressão",
}

LEVEL_DESCRIPTIONS = {
    1: "Fatos atômicos: +/−/× de 1 dígito para velocidade e automatização.",
    2: "Operações em 2 passos: aritmética de 2 dígitos e pequenas cadeias (decomposição).",
    3: "Fluxo de total corrente: mantenha um total ao longo de uma sequência (memória de trabalho).",
    4: "Saltos multiplicativos: fluxos com × e ÷ (alta carga).",
    5: "Modo compressão: expressões densas mistas sob pressão de tempo (elite).",
}

DEFAULT_TIME_LIMITS = {1: 3.0, 2: 7.0, 3: 10.0, 4: 12.0, 5: 12.0}
DEFAULT_ITEMS_PER_ROUND = {3: 6, 4: 4}
DIFF_WEIGHTS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}

DB_PATH = Path("rnf_leaderboard.sqlite3")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def now_ms() -> int:
    return int(time.time() * 1000)


def format_seconds(x: float) -> str:
    return f"{x:.2f}s"


def safe_int(s: str) -> Optional[int]:
    try:
        s = s.strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


@dataclass
class Attempt:
    ts_ms: int
    level: int
    prompt: str
    correct: int
    answer: Optional[int]
    is_correct: bool
    rt_s: float
    time_limit_s: float
    difficulty: float
    score: float


def init_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "landing"
    if "active_level" not in st.session_state:
        st.session_state.active_level = 1
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = 1.0
    if "time_limit_s" not in st.session_state:
        st.session_state.time_limit_s = DEFAULT_TIME_LIMITS[st.session_state.active_level]
    if "attempts" not in st.session_state:
        st.session_state.attempts: List[Attempt] = []
    if "current" not in st.session_state:
        st.session_state.current = None
        st.session_state.pending_generate = True
    if "q_started_ms" not in st.session_state:
        st.session_state.q_started_ms = None
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = None
    if "streak" not in st.session_state:
        st.session_state.streak = 0
    if "best_streak" not in st.session_state:
        st.session_state.best_streak = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = now_ms()
    if "auto_next" not in st.session_state:
        st.session_state.auto_next = True

    # --- Game control ---
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "question_nonce" not in st.session_state:
        st.session_state.question_nonce = 0
    if "pending_generate" not in st.session_state:
        st.session_state.pending_generate = False

    # One-tap (default for Android)
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = T["ONE_TAP"]
    if "choices_n" not in st.session_state:
        st.session_state.choices_n = 4

    # Leaderboard submit state
    if "pending_top_submit" not in st.session_state:
        st.session_state.pending_top_submit = False
    if "alias_input" not in st.session_state:
        st.session_state.alias_input = ""


def difficulty_scale(d: float) -> float:
    return 1.0 + math.log1p(max(0.0, d)) * 0.9


def gen_level_1(d: float) -> Tuple[str, int]:
    scale = difficulty_scale(d)
    max_digit = int(clamp(6 + scale * 2.5, 7, 12))
    op = random.choices(["+", "-", "×"], weights=[2, 2, 3])[0]
    a = random.randint(2, max_digit)
    b = random.randint(2, max_digit)

    if op == "+":
        return f"{a} + {b}", a + b
    if op == "-":
        if d < 4:
            a, b = max(a, b), min(a, b)
        return f"{a} − {b}", a - b

    # ×
    if d < 3:
        b = random.randint(2, min(9, max_digit))
    return f"{a} × {b}", a * b


def gen_level_2(d: float) -> Tuple[str, int]:
    scale = difficulty_scale(d)
    max_two = int(clamp(40 + scale * 25, 60, 180))
    op = random.choices(["+", "-", "×"], weights=[4, 3, 3])[0]

    if op in ["+", "-"]:
        a = random.randint(10, max_two)
        b = random.randint(10, max_two)
        if op == "-" and d < 4:
            a, b = max(a, b), min(a, b)
        return f"{a} {'+' if op=='+' else '−'} {b}", (a + b if op == "+" else a - b)

    a = random.randint(12, max_two)
    b = random.randint(2, int(clamp(6 + scale * 2, 7, 14)))
    base = a * b
    if d < 4:
        return f"{a} × {b}", base

    tweak = random.randint(5, int(clamp(12 + scale * 6, 18, 60)))
    op2 = random.choice(["+", "−"])
    return f"{a} × {b} {op2} {tweak}", (base + tweak if op2 == "+" else base - tweak)


def gen_stream_terms(d: float, count: int, allow_mult: bool) -> List[Tuple[str, int]]:
    scale = difficulty_scale(d)
    terms = []
    for i in range(count):
        if allow_mult and i > 0 and d > 4 and random.random() < 0.25:
            k = random.randint(2, int(clamp(3 + scale * 1.5, 4, 9)))
            op = random.choice(["×", "÷"])
            terms.append((f"{op} {k}", ("MUL", op, k)))
        else:
            mag = int(clamp(8 + scale * 10, 12, 120))
            val = random.randint(2, mag)
            sign = random.choice(["+", "−"])
            terms.append((f"{sign} {val}", val if sign == "+" else -val))
    return terms


def eval_stream(start: int, terms: List[Tuple[str, int]]) -> int:
    total = start
    for _, delta in terms:
        if isinstance(delta, tuple) and delta[0] == "MUL":
            _, op, k = delta
            if op == "×":
                total *= k
            else:
                if total % k != 0:
                    total -= (total % k)
                total //= k
        else:
            total += int(delta)
    return total


def gen_level_3(d: float, count: int) -> Tuple[str, int]:
    scale = difficulty_scale(d)
    start_mag = int(clamp(0 + scale * 12, 10, 80))
    start = 0 if d < 2.5 else random.randint(-start_mag, start_mag)
    terms = gen_stream_terms(d, count=count, allow_mult=False)
    prompt = "Start: " + str(start) + "\n" + "\n".join([t for t, _ in terms]) + "\n= ?"
    return prompt, eval_stream(start, terms)


def gen_level_4(d: float, count: int) -> Tuple[str, int]:
    scale = difficulty_scale(d)
    start = random.randint(5, int(clamp(10 + scale * 15, 20, 120)))
    terms = gen_stream_terms(d, count=count, allow_mult=True)
    prompt = "Start: " + str(start) + "\n" + "\n".join([t for t, _ in terms]) + "\n= ?"
    return prompt, eval_stream(start, terms)


def gen_level_5(d: float) -> Tuple[str, int]:
    scale = difficulty_scale(d)
    a = random.randint(20, int(clamp(40 + scale * 40, 70, 280)))
    b = random.randint(2, int(clamp(6 + scale * 2.2, 8, 18)))
    c = random.randint(10, int(clamp(20 + scale * 18, 40, 120)))
    d2 = random.randint(5, int(clamp(12 + scale * 10, 20, 100)))
    e = random.randint(3, int(clamp(10 + scale * 5, 14, 60)))

    op1 = random.choice(["−", "+"])
    op2 = random.choice(["−", "+"])
    base = a * b
    expr = f"{a} × {b} {op1} {c} {op2} {d2}"
    ans = base - c if op1 == "−" else base + c
    ans = ans - d2 if op2 == "−" else ans + d2

    if d > 5 and random.random() < 0.5:
        k = random.randint(2, int(clamp(3 + scale * 1.2, 4, 10)))
        op3 = random.choice(["−", "+"])
        expr += f" {op3} {e} × {k}"
        ans = ans - (e * k) if op3 == "−" else ans + (e * k)

    return expr, ans


def generate_question(level: int, difficulty: float, stream_len_3: int, stream_len_4: int) -> Dict:
    if level == 1:
        prompt, correct = gen_level_1(difficulty)
    elif level == 2:
        prompt, correct = gen_level_2(difficulty)
    elif level == 3:
        prompt, correct = gen_level_3(difficulty, count=stream_len_3)
    elif level == 4:
        prompt, correct = gen_level_4(difficulty, count=stream_len_4)
    elif level == 5:
        prompt, correct = gen_level_5(difficulty)
    else:
        raise ValueError("Invalid level")
    return {"prompt": prompt, "correct": correct}


def update_difficulty(difficulty: float, is_correct: bool, rt_s: float, time_limit_s: float) -> float:
    if not is_correct:
        difficulty -= 0.45
    else:
        speed_ratio = rt_s / max(0.001, time_limit_s)
        if speed_ratio < 0.45:
            difficulty += 0.40
        elif speed_ratio < 0.70:
            difficulty += 0.20
        elif speed_ratio < 0.95:
            difficulty += 0.05
        else:
            difficulty -= 0.05
    return clamp(difficulty, 0.5, 30.0)


def compute_score(is_correct: bool, rt_s: float, level: int, difficulty: float) -> float:
    if not is_correct:
        return 0.0
    base = DIFF_WEIGHTS[level] * (1.0 + 0.06 * difficulty)
    return base / max(0.25, rt_s)


def summary_stats(attempts: List[Attempt]) -> Dict[str, float]:
    if not attempts:
        return {"n": 0, "acc": 0.0, "avg_rt": 0.0, "score": 0.0}
    n = len(attempts)
    correct = sum(1 for a in attempts if a.is_correct)
    acc = correct / n
    avg_rt = sum(a.rt_s for a in attempts) / n
    score = sum(a.score for a in attempts)
    return {"n": n, "acc": acc, "avg_rt": avg_rt, "score": score}


def stats_for_level(attempts: List[Attempt], level: int) -> Dict[str, float]:
    return summary_stats([a for a in attempts if a.level == level])


def export_csv(attempts: List[Attempt]) -> bytes:
    out = io.StringIO()
    if not attempts:
        return b""
    writer = csv.DictWriter(out, fieldnames=list(asdict(attempts[0]).keys()))
    writer.writeheader()
    for a in attempts:
        writer.writerow(asdict(a))
    return out.getvalue().encode("utf-8")


def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER NOT NULL,
            alias TEXT NOT NULL,
            score REAL NOT NULL,
            problems INTEGER NOT NULL,
            acc REAL NOT NULL,
            avg_rt REAL NOT NULL,
            created_ts_ms INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_top_scores(level: int, limit: int = 10):
    conn = db_conn()
    cur = conn.execute(
        "SELECT alias, score, problems, acc, avg_rt, created_ts_ms "
        "FROM leaderboard WHERE level = ? "
        "ORDER BY score DESC, problems DESC, created_ts_ms ASC LIMIT ?",
        (int(level), int(limit)),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_best_score(level: int) -> float:
    conn = db_conn()
    cur = conn.execute("SELECT COALESCE(MAX(score), 0) FROM leaderboard WHERE level = ?", (int(level),))
    best = float(cur.fetchone()[0] or 0.0)
    conn.close()
    return best


def submit_score(level: int, alias: str, score: float, problems: int, acc: float, avg_rt: float):
    alias = (alias or "").strip()
    if not alias:
        raise ValueError("Apelido vazio.")
    alias = alias[:24]
    conn = db_conn()
    conn.execute(
        "INSERT INTO leaderboard(level, alias, score, problems, acc, avg_rt, created_ts_ms) "
        "VALUES(?,?,?,?,?,?,?)",
        (int(level), alias, float(score), int(problems), float(acc), float(avg_rt), now_ms()),
    )
    conn.commit()
    conn.close()


def reset_level_attempts(level: int):
    st.session_state.attempts = [a for a in st.session_state.attempts if a.level != level]
    st.session_state.current = None
        st.session_state.pending_generate = True
    st.session_state.q_started_ms = None
    st.session_state.last_feedback = None
    st.session_state.streak = 0
    st.session_state.best_streak = 0


def make_choices(correct: int, level: int, n: int) -> List[int]:
    rng = random.Random()
    choices = {correct}

    base_steps = {
        1: [1, 2, 3, 5, 10],
        2: [1, 2, 5, 10, 20],
        3: [1, 2, 5, 10, 20, 50],
        4: [1, 2, 5, 10, 25, 50, 100],
        5: [1, 2, 5, 10, 25, 50, 100, 200],
    }[level]

    attempts = 0
    while len(choices) < n and attempts < 500:
        attempts += 1
        step = rng.choice(base_steps)
        variant = rng.random()

        if variant < 0.60:
            cand = correct + rng.choice([-step, step])
        elif variant < 0.75:
            cand = correct + rng.choice([-1, 1]) * (step + 1)
        elif variant < 0.88:
            cand = -correct + rng.choice([-step, step])
        else:
            k = rng.choice([1, 2])
            cand = correct + rng.choice([-1, 1]) * (10**k)

        if cand != correct:
            choices.add(int(cand))

    while len(choices) < n:
        choices.add(int(correct + rng.randint(-50, 50)))

    out = list(choices)
    rng.shuffle(out)
    return out[:n]


def one_tap_panel(choices: List[int]) -> Optional[int]:
    cols = st.columns(2)
    picked = None
    for i, val in enumerate(choices):
        with cols[i % 2]:
            if st.button(str(val), use_container_width=True, key=f"choice_{st.session_state.question_nonce}_{i}"):
                picked = val
    return picked


def landing_screen():
    if not st.session_state.game_started:
    if st.button("Start Game" if "Rapid Number Forge" in st.session_state.get("title","") else "Iniciar Jogo", type="primary"):
        st.session_state.game_started = True
        st.session_state.pending_generate = True
        st.rerun()

st.subheader(T["CHOOSE_LEVEL"])
    st.write(T["PICK_TODAY"])

    for lvl in LEVELS.keys():
        with st.container(border=True):
            st.markdown(f"### Nível {lvl} — {LEVELS[lvl]}")
            st.write(LEVEL_DESCRIPTIONS.get(lvl, ""))

            cols = st.columns([1, 2, 2])
            with cols[0]:
                if st.button(f"{T['START_LEVEL']} {lvl}", key=f"start_{lvl}", type="primary"):
                    st.session_state.active_level = lvl
                    st.session_state.time_limit_s = DEFAULT_TIME_LIMITS[lvl]
                    st.session_state.current = None
        st.session_state.pending_generate = True
                    st.session_state.q_started_ms = None
                    st.session_state.last_feedback = None
                    st.session_state.streak = 0
                    st.session_state.mode = "play"
                    st.rerun()

            with cols[1]:
                st.caption(f"{T['DEFAULT_TIME']} {DEFAULT_TIME_LIMITS[lvl]:.1f}s")

            with cols[2]:
                lvl_stats = stats_for_level(st.session_state.attempts, lvl)
                st.caption(f"{T['YOUR_SESSION_SCORE'].format(lvl=lvl)} {lvl_stats['score']:.2f}")

    st.divider()
    st.caption(T["TIP_LANDING"])


def sidebar_controls():
    st.sidebar.header(T["SETTINGS"])

    level = st.sidebar.selectbox(
        T["LEVEL"],
        options=list(LEVELS.keys()),
        format_func=lambda k: f"{k} — {LEVELS[k]}",
        index=list(LEVELS.keys()).index(st.session_state.active_level),
    )
    if level != st.session_state.active_level:
        st.session_state.active_level = level
        st.session_state.time_limit_s = DEFAULT_TIME_LIMITS[level]
        st.session_state.current = None
        st.session_state.pending_generate = True
        st.session_state.q_started_ms = None
        st.session_state.last_feedback = None

    st.session_state.auto_next = st.sidebar.toggle(T["AUTO_NEXT"], value=st.session_state.auto_next)

    st.session_state.input_mode = st.sidebar.selectbox(
        T["INPUT_MODE"],
        options=[T["ONE_TAP"], T["TYPING"]],
        index=0 if st.session_state.input_mode == T["ONE_TAP"] else 1,
    )
    st.session_state.choices_n = st.sidebar.slider(T["CHOICES_PER"], 3, 8, int(st.session_state.choices_n), 1)

    st.session_state.time_limit_s = st.sidebar.slider(T["TIME_LIMIT"], 1.5, 20.0, float(st.session_state.time_limit_s), 0.5)

    stream_len_3 = st.sidebar.slider(T["L3_STREAM"], 3, 14, DEFAULT_ITEMS_PER_ROUND[3], 1)
    stream_len_4 = st.sidebar.slider(T["L4_STEPS"], 3, 10, DEFAULT_ITEMS_PER_ROUND[4], 1)

    st.sidebar.divider()
    st.sidebar.subheader(T["CALIBRATION"])
    st.sidebar.caption(T["CALIBRATION_CAP"])
    st.session_state.difficulty = st.sidebar.slider(T["DIFFICULTY"], 0.5, 30.0, float(st.session_state.difficulty), 0.5)

    st.sidebar.divider()
    st.sidebar.subheader(T["LEVEL_TOOLS"])
    if st.sidebar.button(T["RESET_LEVEL"], type="secondary"):
        reset_level_attempts(st.session_state.active_level)
        st.sidebar.success(T["RESET_DONE"])
        st.rerun()

    if st.sidebar.button(T["BACK_CHOOSER"], type="secondary"):
        st.session_state.mode = "landing"
        st.session_state.current = None
        st.session_state.pending_generate = True
        st.session_state.q_started_ms = None
        st.session_state.last_feedback = None
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button(T["RESET_SESSION"], type="secondary"):
        st.session_state.attempts = []
        st.session_state.streak = 0
        st.session_state.best_streak = 0
        st.session_state.current = None
        st.session_state.pending_generate = True
        st.session_state.q_started_ms = None
        st.session_state.last_feedback = None
        st.session_state.session_id = now_ms()
        st.session_state.pending_top_submit = False
        st.session_state.alias_input = ""
        st.rerun()

    return stream_len_3, stream_len_4


def ensure_question(stream_len_3: int, stream_len_4: int):
    # Only generate if game started
    if not st.session_state.game_started:
        return
    if (st.session_state.current is None) or st.session_state.pending_generate:
        st.session_state.current = generate_question(
            st.session_state.active_level,
            st.session_state.difficulty,
            stream_len_3=stream_len_3,
            stream_len_4=stream_len_4,
        )
        st.session_state.q_started_ms = now_ms()
        st.session_state.question_nonce += 1
        st.session_state.pending_generate = False


def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    init_state()

    st.title(APP_TITLE)
    st.caption(T["TAGLINE"])

    if st.session_state.mode == "landing":
        landing_screen()
        return

    stream_len_3, stream_len_4 = sidebar_controls()
    ensure_question(stream_len_3, stream_len_4)

    stats_all = summary_stats(st.session_state.attempts)
    stats_lvl = stats_for_level(st.session_state.attempts, st.session_state.active_level)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["LEVEL"], f"{st.session_state.active_level} — {LEVELS[st.session_state.active_level]}")
    c2.metric("Acc", f"{stats_lvl['acc']*100:.1f}%")
    c3.metric("Avg RT", format_seconds(stats_lvl["avg_rt"]) if stats_lvl["n"] else "—")
    c4.metric("Streak", f"{st.session_state.streak} (best {st.session_state.best_streak})")

    st.divider()

    prompt = st.session_state.current["prompt"]
    correct = st.session_state.current["correct"]

    st.subheader(T["PROBLEM"])
    st.code(prompt, language="text")
    st.info(T["INFO_BAR"].format(tl=st.session_state.time_limit_s, diff=st.session_state.difficulty, score=stats_lvl["score"]))

    submitted = False
    answer = None

    if st.session_state.input_mode == T["ONE_TAP"]:
        choices = make_choices(int(correct), st.session_state.active_level, int(st.session_state.choices_n))
        picked = one_tap_panel(choices)
        if picked is not None:
            submitted = True
            answer = int(picked)
    else:
        ans_str = st.text_input(T["ANSWER"], placeholder="ex.: 42 ou -7")
        if st.button(T["SUBMIT"], type="primary"):
            submitted = True
            answer = safe_int(ans_str)

    if submitted:
        t_end_ms = now_ms()
        rt_s = (t_end_ms - st.session_state.q_started_ms) / 1000.0 if st.session_state.q_started_ms else 0.0

        timed_out = rt_s > st.session_state.time_limit_s
        is_correct = (answer == correct) and (not timed_out)

        if is_correct:
            st.session_state.streak += 1
            st.session_state.best_streak = max(st.session_state.best_streak, st.session_state.streak)
        else:
            st.session_state.streak = 0

        old_d = st.session_state.difficulty
        new_d = update_difficulty(old_d, is_correct, rt_s, st.session_state.time_limit_s)
        st.session_state.difficulty = new_d

        score = compute_score(is_correct, rt_s, st.session_state.active_level, old_d)

        st.session_state.attempts.append(
            Attempt(
                ts_ms=t_end_ms,
                level=st.session_state.active_level,
                prompt=prompt.replace("\n", " | "),
                correct=int(correct),
                answer=answer,
                is_correct=bool(is_correct),
                rt_s=float(rt_s),
                time_limit_s=float(st.session_state.time_limit_s),
                difficulty=float(old_d),
                score=float(score),
            )
        )

        if answer is None:
            st.session_state.last_feedback = ("error", T["ENTER_INT"].format(correct=correct))
        elif timed_out:
            st.session_state.last_feedback = ("error", T["TIMEOUT"].format(rt=format_seconds(rt_s), correct=correct))
        elif is_correct:
            st.session_state.last_feedback = ("success", T["CORRECT"].format(rt=format_seconds(rt_s), score=score, new_d=new_d))
        else:
            st.session_state.last_feedback = ("error", T["INCORRECT"].format(answer=answer, rt=format_seconds(rt_s), correct=correct, new_d=new_d))

        st.session_state.current = None
        st.session_state.pending_generate = True
        st.session_state.q_started_ms = None
        if st.session_state.auto_next:
            ensure_question(stream_len_3, stream_len_4)
        st.rerun()

    if st.session_state.last_feedback:
        kind, msg = st.session_state.last_feedback
        st.success(msg) if kind == "success" else st.error(msg)

    st.divider()
    st.subheader(T["SESSION_SUMMARY"])

    colA, colB, colC = st.columns(3)
    colA.metric(T["PROBLEMS_ALL"], f"{stats_all['n']}")
    colB.metric(T["TOTAL_SCORE_ALL"], f"{stats_all['score']:.2f}")
    colC.metric(T["SCORE_LEVEL"], f"{stats_lvl['score']:.2f}")

    if st.session_state.attempts:
        with st.expander(T["HISTORY"]):
            for a in reversed(st.session_state.attempts[-60:]):
                icon = "✅" if a.is_correct else "❌"
                st.write(
                    f"{icon} L{a.level} • RT {format_seconds(a.rt_s)} / {a.time_limit_s:.1f}s • "
                    f"Resp {a.answer} • Correto {a.correct} • Score {a.score:.2f}"
                )
                st.caption(a.prompt)

        st.download_button(
            label=T["DOWNLOAD_CSV"],
            data=export_csv(st.session_state.attempts),
            file_name=f"rnf_session_{st.session_state.session_id}.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader(T["LEADERBOARD_TITLE"])

    lb_level = st.selectbox(
        T["LEADERBOARD_LEVEL"],
        options=list(LEVELS.keys()),
        format_func=lambda k: f"Nível {k} — {LEVELS[k]}",
        index=list(LEVELS.keys()).index(st.session_state.active_level),
        key="leaderboard_level_select",
    )

    top = get_top_scores(level=lb_level, limit=10)
    if not top:
        st.info(T["NO_SCORES"])
    else:
        for i, (alias, score_v, problems, acc, avg_rt, ts_ms) in enumerate(top, start=1):
            st.write(f"**#{i}** — {alias} • **{score_v:.2f}** pts • {problems} problemas • {acc*100:.1f}% acc • {avg_rt:.2f}s RT médio")

    MIN_PROBLEMS_FOR_LEADERBOARD = 25

    active_lvl = st.session_state.active_level
    s = stats_for_level(st.session_state.attempts, active_lvl)
    active_best = get_best_score(level=active_lvl)

    if s["n"] >= MIN_PROBLEMS_FOR_LEADERBOARD and s["score"] > active_best:
        st.success(T["NEW_TOP"].format(lvl=active_lvl))
        st.session_state.pending_top_submit = True

    if st.session_state.pending_top_submit:
        with st.form("leaderboard_submit"):
            alias = st.text_input(T["ALIAS"], value=st.session_state.alias_input, max_chars=24)
            submitted_alias = st.form_submit_button(T["SUBMIT_LEVEL"].format(lvl=active_lvl))
        if submitted_alias:
            try:
                submit_score(level=active_lvl, alias=alias, score=s["score"], problems=s["n"], acc=s["acc"], avg_rt=s["avg_rt"])
                st.session_state.pending_top_submit = False
                st.session_state.alias_input = ""
                st.success(T["SUBMITTED"])
                st.rerun()
            except Exception as e:
                st.error(T["SUBMIT_FAIL"].format(e=e))

    st.caption(T["CONSISTENCY_TIP"])


if __name__ == "__main__":
    main()
