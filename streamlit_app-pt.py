
import streamlit as st
import random
import time
import matplotlib.pyplot as plt

# ============================================================
# MATH WIZARD — SINGLE FILE STREAMLIT APP
# Pages: Game | Profile / Stats
# Features:
# - Per-game-type levels (current + unlocked max)
# - Per-game-type freeze level toggle
# - User can move back current level to any unlocked level
# - Warm-up after time away (not counted in stats, quick)
# - Accuracy + Response Time (RT) evolution graphs (Profile page only)
# - Reset/Start-over button (Profile page only)
# ============================================================

MODES = ["Addition", "Multiplication"]

# ==============================
# SESSION STATE INITIALIZATION
# ==============================

def _init_dict(key: str, default_factory):
    if key not in st.session_state:
        st.session_state[key] = {m: default_factory(m) for m in MODES}

def _init_simple(key: str, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

_init_simple("mode", "Addition")

# Per-mode progression state
_init_dict("current_level", lambda _: 1)
_init_dict("unlocked_level", lambda _: 1)
_init_dict("difficulty_stage", lambda _: "I")
_init_dict("memory_load", lambda _: 1)
_init_dict("freeze_level", lambda _: False)

# Per-mode performance stats
_init_dict("score", lambda _: 0)
_init_dict("streak", lambda _: 0)
_init_dict("total_questions", lambda _: 0)
_init_dict("correct_answers", lambda _: 0)

# Warm-up management (per mode)
_init_dict("warmup_remaining", lambda _: 0)

# Measured history logs (per mode)
# Each entry: {"t": float, "correct": bool, "rt": float, "level": int, "score_delta": int}
_init_dict("history", lambda _: [])

# Question state
_init_simple("question", "")
_init_simple("correct_result", 0)
_init_simple("question_mode", st.session_state.mode)  # the mode this question was generated for
_init_dict("question_start_ts", lambda _: time.time())

# Activity time (to trigger warm-up after being away)
_init_simple("last_active_ts", time.time())
_init_simple("away_warmup_armed", False)

# ==============================
# CONSTANTS
# ==============================

WARMUP_THRESHOLD_SECONDS = 6 * 60 * 60  # 6 hours
WARMUP_QUESTIONS = 3                   # "Quick"
STREAK_TO_LEVEL_UP = 5

STAGES = ["I", "II", "III", "IV", "V", "VI", "VII"]

# ==============================
# HELPERS
# ==============================

def stage_for_level(level: int) -> str:
    idx = min((max(level, 1) - 1) // 3, len(STAGES) - 1)
    return STAGES[idx]

def arm_warmup_if_away():
    now = time.time()
    away_for = now - st.session_state.last_active_ts
    if away_for >= WARMUP_THRESHOLD_SECONDS and not st.session_state.away_warmup_armed:
        for m in MODES:
            st.session_state.warmup_remaining[m] = WARMUP_QUESTIONS
        st.session_state.away_warmup_armed = True

def mark_active_now():
    st.session_state.last_active_ts = time.time()
    st.session_state.away_warmup_armed = False

def update_level_if_allowed(mode: str):
    if st.session_state.freeze_level[mode]:
        return

    if st.session_state.streak[mode] >= STREAK_TO_LEVEL_UP:
        st.session_state.streak[mode] = 0
        st.session_state.current_level[mode] += 1

        if st.session_state.current_level[mode] > st.session_state.unlocked_level[mode]:
            st.session_state.unlocked_level[mode] = st.session_state.current_level[mode]

        st.session_state.memory_load[mode] += 1
        st.session_state.difficulty_stage[mode] = stage_for_level(st.session_state.current_level[mode])

def clamp_current_level_to_unlocked(mode: str):
    cur = st.session_state.current_level[mode]
    unlocked = st.session_state.unlocked_level[mode]
    if cur < 1:
        st.session_state.current_level[mode] = 1
    elif cur > unlocked:
        st.session_state.current_level[mode] = unlocked

def warmup_active(mode: str) -> bool:
    return st.session_state.warmup_remaining[mode] > 0

def reset_all_data():
    keep = {"mode"}
    keys = list(st.session_state.keys())
    for k in keys:
        if k not in keep:
            del st.session_state[k]
    st.rerun()

# ==============================
# QUESTION GENERATORS
# ==============================

def generate_addition(level: int):
    if level == 1:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
    elif level == 2:
        a = random.randint(10, 99)
        b = random.randint(0, 9)
    else:
        digits = min(2 + (level // 2), 7)
        a = random.randint(10 ** (digits - 1), 10 ** digits - 1)
        b = random.randint(10 ** (digits - 1), 10 ** digits - 1)

    return f"{a} + {b}", a + b

def generate_multiplication(level: int):
    if level == 1:
        a = random.randint(1, 5)
        b = random.randint(1, 5)
    elif level == 2:
        a = random.randint(2, 9)
        b = random.randint(2, 9)
    else:
        digits = min(1 + (level // 3), 4)
        a = random.randint(10 ** (digits - 1), 10 ** digits - 1)
        b = random.randint(2, 9)

    return f"{a} × {b}", a * b

def generate_question_for(mode: str, level: int):
    if mode == "Addition":
        return generate_addition(level)
    return generate_multiplication(level)

def new_question():
    mode = st.session_state.mode

    if warmup_active(mode):
        target_level = max(1, st.session_state.current_level[mode] - 1)
    else:
        target_level = st.session_state.current_level[mode]

    q, ans = generate_question_for(mode, target_level)

    st.session_state.question = q
    st.session_state.correct_result = ans
    st.session_state.question_mode = mode
    st.session_state.question_start_ts[mode] = time.time()

# ==============================
# GRAPH HELPERS
# ==============================

def _rolling(values, window: int):
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def plot_metric(metric: str, subtype: str, mode_choice: str):
    modes_to_plot = MODES if mode_choice == "Both" else [mode_choice]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    any_data = False

    for m in modes_to_plot:
        hist = st.session_state.history[m]
        if not hist:
            continue

        any_data = True

        if metric == "Accuracy":
            correct_flags = [1 if h["correct"] else 0 for h in hist]
            x = list(range(1, len(correct_flags) + 1))

            if subtype == "Cumulative":
                y = []
                c = 0
                for i, v in enumerate(correct_flags, start=1):
                    c += v
                    y.append((c / i) * 100.0)
                ax.plot(x, y, label=m)

            elif subtype == "Rolling (20)":
                y = _rolling(correct_flags, 20)
                y = [v * 100.0 for v in y]
                ax.plot(x, y, label=m)

            elif subtype == "By Level (avg %)":
                level_map = {}
                for h in hist:
                    lvl = h["level"]
                    level_map.setdefault(lvl, []).append(1 if h["correct"] else 0)
                lvls = sorted(level_map.keys())
                y = [(sum(level_map[l]) / len(level_map[l])) * 100.0 for l in lvls]
                ax.plot(lvls, y, marker="o", label=m)
                ax.set_xlabel("Level")
                ax.set_title("Accuracy by Level")
                ax.set_ylabel("Accuracy (%)")

        elif metric == "Response Time":
            # Use only non-None RTs
            seq = [h.get("rt") for h in hist if h.get("rt") is not None]
            if not seq:
                continue

            if subtype == "Cumulative avg (s)":
                y = []
                s = 0.0
                for i, rt in enumerate(seq, start=1):
                    s += rt
                    y.append(s / i)
                x = list(range(1, len(y) + 1))
                ax.plot(x, y, label=m)

            elif subtype == "Rolling (20) (s)":
                y = _rolling(seq, 20)
                x = list(range(1, len(y) + 1))
                ax.plot(x, y, label=m)

            elif subtype == "By Level (median s)":
                level_map = {}
                for h in hist:
                    rt = h.get("rt")
                    if rt is None:
                        continue
                    lvl = h["level"]
                    level_map.setdefault(lvl, []).append(rt)
                lvls = sorted(level_map.keys())
                y = []
                for l in lvls:
                    vals = sorted(level_map[l])
                    mid = len(vals) // 2
                    if len(vals) % 2 == 1:
                        y.append(vals[mid])
                    else:
                        y.append((vals[mid - 1] + vals[mid]) / 2.0)
                ax.plot(lvls, y, marker="o", label=m)
                ax.set_xlabel("Level")
                ax.set_title("Response Time by Level")
                ax.set_ylabel("Seconds")

        elif metric == "Score":
            deltas = [h.get("score_delta", 0) for h in hist]
            x = list(range(1, len(deltas) + 1))

            if subtype == "Cumulative":
                y = []
                s = 0
                for d in deltas:
                    s += d
                    y.append(s)
                ax.plot(x, y, label=m)

            elif subtype == "Per Question (delta)":
                ax.plot(x, deltas, label=m)

            elif subtype == "By Level (avg / q)":
                level_map = {}
                for h in hist:
                    lvl = h["level"]
                    level_map.setdefault(lvl, []).append(h.get("score_delta", 0))
                lvls = sorted(level_map.keys())
                y = [sum(level_map[l]) / len(level_map[l]) for l in lvls]
                ax.plot(lvls, y, marker="o", label=m)
                ax.set_xlabel("Level")
                ax.set_title("Average Score per Question by Level")
                ax.set_ylabel("Points")

    if not any_data:
        return None

    if ax.get_title() == "":
        ax.set_title(f"{metric} — {subtype}")
    if ax.get_xlabel() == "":
        ax.set_xlabel("Question #")

    if metric == "Accuracy" and subtype in ("Cumulative", "Rolling (20)"):
        ax.set_ylabel("Accuracy (%)")
    if metric == "Response Time" and subtype in ("Cumulative avg (s)", "Rolling (20) (s)"):
        ax.set_ylabel("Seconds")
    if metric == "Score" and subtype in ("Cumulative", "Per Question (delta)"):
        ax.set_ylabel("Points")

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    return fig

# ==============================
# SIDEBAR NAVIGATION
# ==============================

arm_warmup_if_away()

st.sidebar.title("MATH WIZARD")
page = st.sidebar.radio("Navigate", ["Game", "Profile / Stats"])
st.sidebar.selectbox("Game Mode", MODES, key="mode")

clamp_current_level_to_unlocked(st.session_state.mode)

# ==============================
# PAGE: GAME
# ==============================

if page == "Game":
    mode = st.session_state.mode

    st.title("⚡ Math Wizard")
    st.write(f"### Mode: {mode}")

    if warmup_active(mode):
        st.info(
            f"Warm-up: {st.session_state.warmup_remaining[mode]} question(s) left. "
            f"These do **not** count in your stats."
        )

    st.write(f"Stage: **{st.session_state.difficulty_stage[mode]}**")
    st.write(
        f"Level: **{st.session_state.current_level[mode]}** "
        f"(Unlocked: **{st.session_state.unlocked_level[mode]}**)"
    )

    if st.session_state.question == "" or st.session_state.question_mode != mode:
        new_question()

    st.write(f"## {st.session_state.question}")

    user_answer = st.text_input("Your answer", key="answer_input")

    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Submit")
    with col2:
        skip = st.button("Skip (no penalty)")

    if skip:
        mark_active_now()
        st.session_state.answer_input = ""
        new_question()
        st.rerun()

    if submit:
        raw = user_answer.strip()

        if raw == "":
            st.warning("Type an answer first.")
        else:
            try:
                user_int = int(raw)
            except ValueError:
                st.error("Please enter a whole number.")
                st.stop()

            mark_active_now()
            is_warmup = warmup_active(mode)

            # Compute RT (seconds)
            rt = None
            try:
                rt = max(0.0, time.time() - float(st.session_state.question_start_ts.get(mode, time.time())))
            except Exception:
                rt = None

            if is_warmup:
                if user_int == st.session_state.correct_result:
                    st.success("Correct (warm-up).")
                else:
                    st.error(f"Wrong (warm-up). Correct answer: {st.session_state.correct_result}")

                st.session_state.warmup_remaining[mode] = max(0, st.session_state.warmup_remaining[mode] - 1)

                st.session_state.answer_input = ""
                new_question()
                st.rerun()

            # Normal measured question
            st.session_state.total_questions[mode] += 1
            correct = (user_int == st.session_state.correct_result)

            score_delta = 0
            if correct:
                st.success("Correct!")
                score_delta = 10
                st.session_state.score[mode] += score_delta
                st.session_state.streak[mode] += 1
                st.session_state.correct_answers[mode] += 1
                update_level_if_allowed(mode)
            else:
                st.error(f"Wrong! Correct answer: {st.session_state.correct_result}")
                st.session_state.streak[mode] = 0

            # History log (measured only)
            st.session_state.history[mode].append(
                {
                    "t": time.time(),
                    "correct": bool(correct),
                    "rt": rt,
                    "level": int(st.session_state.current_level[mode]),
                    "score_delta": int(score_delta),
                }
            )

            st.session_state.answer_input = ""
            new_question()
            st.session_state.question_start_ts[mode] = time.time()
            st.rerun()

# ==============================
# PAGE: PROFILE / STATS
# ==============================

if page == "Profile / Stats":
    st.title("📊 Profile / Stats")

    st.subheader("Per-Mode Controls (Level + Freeze)")

    for mode in MODES:
        st.markdown(f"#### {mode}")

        unlocked = st.session_state.unlocked_level[mode]
        level_options = list(range(1, unlocked + 1))

        selected = st.selectbox(
            f"Current level for {mode} (choose any unlocked level)",
            options=level_options,
            index=level_options.index(st.session_state.current_level[mode]) if st.session_state.current_level[mode] in level_options else len(level_options) - 1,
            key=f"level_picker_{mode}"
        )
        st.session_state.current_level[mode] = selected
        st.session_state.difficulty_stage[mode] = stage_for_level(st.session_state.current_level[mode])

        st.session_state.freeze_level[mode] = st.checkbox(
            f"Freeze level progression for {mode}",
            value=st.session_state.freeze_level[mode],
            key=f"freeze_{mode}"
        )

        st.write(f"Unlocked level: **{st.session_state.unlocked_level[mode]}**")
        st.write(f"Difficulty stage: **{st.session_state.difficulty_stage[mode]}**")
        st.write(f"Memory load: **{st.session_state.memory_load[mode]}**")
        st.divider()

    st.subheader("Performance (Measured Only)")

    for mode in MODES:
        total = st.session_state.total_questions[mode]
        correct = st.session_state.correct_answers[mode]
        accuracy = (correct / total) * 100 if total > 0 else 0.0

        st.markdown(f"#### {mode}")
        st.write(f"Score: **{st.session_state.score[mode]}**")
        st.write(f"Current streak: **{st.session_state.streak[mode]}**")
        st.write(f"Total questions: **{total}**")
        st.write(f"Accuracy: **{accuracy:.2f}%**")
        st.progress(min(max(accuracy / 100, 0.0), 1.0))
        st.divider()

    st.subheader("Evolution Graphs")

    metric_type = st.selectbox(
        "Graph type",
        ["Accuracy", "Response Time", "Score"],
        key="graph_metric_type"
    )

    if metric_type == "Accuracy":
        subtype_options = ["Cumulative", "Rolling (20)", "By Level (avg %)"]
    elif metric_type == "Response Time":
        subtype_options = ["Cumulative avg (s)", "Rolling (20) (s)", "By Level (median s)"]
    else:
        subtype_options = ["Cumulative", "Per Question (delta)", "By Level (avg / q)"]

    subtype = st.selectbox(
        "Graph subtype",
        subtype_options,
        key="graph_subtype"
    )

    mode_choice = st.selectbox(
        "Game type to display",
        ["Addition", "Multiplication", "Both"],
        key="graph_mode_choice"
    )

    fig = plot_metric(metric_type, subtype, mode_choice)
    if fig is None:
        st.info("No measured data yet for the selected graph.")
    else:
        st.pyplot(fig)

    st.caption("Warm-up questions (after time away) are excluded from stats and graphs.")

    st.divider()
    st.subheader("Reset / Start Over")

    st.warning("This will reset your progress and statistics (for this session).")
    confirm = st.checkbox("I understand and want to reset everything.", key="reset_confirm")
    if st.button("Reset all data / start over", disabled=not confirm):
        reset_all_data()
