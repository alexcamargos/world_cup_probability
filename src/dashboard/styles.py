"""CSS injection for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --dashboard-bg: #f4f7fb;
            --surface: #ffffff;
            --surface-raised: #fbfcfe;
            --surface-tint: #f7faf9;
            --panel-border: #d9e2ec;
            --panel-border-strong: #c8d4e0;
            --ink: #162033;
            --ink-soft: #344054;
            --muted: #667085;
            --muted-soft: #98a2b3;
            --accent: #08786c;
            --accent-strong: #065f56;
            --accent-soft: #e2f3ef;
            --accent-blue: #2563a8;
            --accent-blue-soft: #e7f0fb;
            --accent-coral: #c85547;
            --accent-coral-soft: #fde9e6;
            --accent-amber: #c78414;
            --accent-amber-soft: #fff3d7;
            --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.06);
            --shadow-md: 0 16px 36px rgba(22, 32, 51, 0.08);
            --radius: 8px;
            --radius-sm: 6px;
        }
        html {
            scroll-behavior: smooth;
        }
        .stApp {
            background:
                linear-gradient(180deg, #eef4f8 0, var(--dashboard-bg) 300px),
                var(--dashboard-bg);
            color: var(--ink);
            font-family:
                Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }
        .stApp [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        .block-container {
            max-width: 1480px;
            padding: 1.15rem 1.6rem 2.75rem;
        }
        section[data-testid="stSidebar"] {
            background: #eef4f8;
            border-right: 1px solid var(--panel-border);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            margin-top: 0.45rem;
            text-transform: uppercase;
            text-wrap: balance;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        .dashboard-hero {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.97), rgba(244, 248, 251, 0.94)),
                linear-gradient(90deg, rgba(8, 120, 108, 0.12), rgba(37, 99, 168, 0.08));
            border: 1px solid rgba(200, 212, 224, 0.9);
            border-radius: var(--radius);
            box-shadow: var(--shadow-md);
            color: var(--ink);
            margin: 0.25rem 0 1.05rem;
            overflow: hidden;
            padding: 1.35rem 1.45rem;
            position: relative;
        }
        .dashboard-hero::before {
            background: linear-gradient(180deg, var(--accent), var(--accent-blue));
            bottom: 0;
            content: "";
            left: 0;
            position: absolute;
            top: 0;
            width: 5px;
        }
        .dashboard-hero .eyebrow {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0.12em;
            margin: 0 0 0.4rem;
            text-transform: uppercase;
        }
        .dashboard-hero h1 {
            color: var(--ink);
            font-size: clamp(1.65rem, 3.3vw, 2.65rem);
            letter-spacing: 0;
            line-height: 1.08;
            margin: 0;
            max-width: 980px;
            text-wrap: balance;
        }
        .dashboard-hero p:last-child {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0.7rem 0 0;
            max-width: 820px;
        }
        h1, h2, h3, h4 {
            color: var(--ink);
            letter-spacing: 0;
            text-wrap: balance;
        }
        p, label, span {
            text-wrap: pretty;
        }
        label,
        div[data-testid="stWidgetLabel"] label {
            color: var(--ink-soft) !important;
            font-weight: 750 !important;
        }
        div[data-testid="stCaptionContainer"] {
            color: var(--muted);
        }
        div[data-testid="stAlert"] {
            border-radius: var(--radius);
            border: 1px solid var(--panel-border);
        }
        div[data-testid="stTextInput"] input,
        div[data-baseweb="select"] > div,
        input,
        textarea,
        select {
            background: var(--surface) !important;
            border-color: var(--panel-border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--ink) !important;
        }
        button,
        [role="button"],
        [role="tab"],
        input,
        textarea,
        select {
            touch-action: manipulation;
        }
        button:focus-visible,
        [role="button"]:focus-visible,
        [role="tab"]:focus-visible,
        input:focus-visible,
        textarea:focus-visible {
            outline: 3px solid rgba(8, 120, 108, 0.28) !important;
            outline-offset: 2px;
        }
        div[data-testid="stButton"] button {
            background: var(--ink);
            border: 1px solid var(--ink);
            border-radius: var(--radius-sm);
            color: #ffffff;
            font-weight: 750;
            min-height: 2.5rem;
            transition:
                background-color 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease,
                transform 140ms ease;
        }
        div[data-testid="stButton"] button:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            box-shadow: 0 8px 18px rgba(8, 120, 108, 0.18);
            transform: translateY(-1px);
        }
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid transparent;
            border-radius: var(--radius-sm);
            padding: 0.1rem 0.25rem;
        }
        div[data-testid="stRadio"] label:hover,
        div[data-testid="stCheckbox"] label:hover {
            border-color: var(--panel-border);
        }
        label[data-baseweb="radio"] > div:first-child {
            background: var(--surface) !important;
            border-color: var(--panel-border-strong) !important;
        }
        label[data-baseweb="radio"]:has(input:checked) > div:first-child {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {
            background: #ffffff !important;
        }
        label[data-baseweb="checkbox"] > span:first-child {
            background: var(--surface) !important;
            border-color: var(--panel-border-strong) !important;
        }
        label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        div[data-testid="stTabs"] {
            margin-top: 0.35rem;
        }
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            gap: 0.2rem;
            overflow-x: auto;
            padding: 0.28rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-border"],
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: transparent !important;
            display: none !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: var(--radius-sm);
            color: var(--muted);
            font-weight: 800;
            min-height: 2.35rem;
            padding: 0.35rem 0.85rem;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            background: var(--surface-tint);
            color: var(--ink);
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: var(--accent-soft);
            color: var(--accent-strong);
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
            color: var(--accent-strong);
        }
        div[data-testid="stMetric"],
        .metric-card,
        .group-card,
        .projection-table-wrap,
        .bracket-board,
        .match-prediction-card,
        .prob-table-wrap {
            background: var(--surface);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
        }
        .metric-grid {
            display: grid;
            gap: 0.85rem;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            margin: 0.85rem 0 1.15rem;
        }
        .metric-grid-four {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .metric-grid-three {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .sidebar-metric-stack {
            display: grid;
            gap: 0.7rem;
            margin-top: 0.8rem;
        }
        .metric-card {
            min-width: 0;
            overflow: hidden;
            padding: 0.9rem 1rem;
            position: relative;
        }
        .metric-card::before {
            background: linear-gradient(180deg, var(--accent), var(--accent-blue));
            bottom: 0;
            content: "";
            left: 0;
            opacity: 0.9;
            position: absolute;
            top: 0;
            width: 3px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 850;
            letter-spacing: 0.02em;
            line-height: 1.25;
            margin-bottom: 0.55rem;
            overflow: hidden;
            overflow-wrap: anywhere;
            text-overflow: clip;
            text-transform: uppercase;
            white-space: normal;
        }
        .metric-value {
            color: var(--ink);
            font-size: clamp(1.45rem, 1.75vw, 1.65rem);
            font-variant-numeric: tabular-nums;
            font-weight: 900;
            line-height: 1.08;
            overflow: hidden;
            overflow-wrap: anywhere;
            text-overflow: clip;
            white-space: normal;
        }
        .metric-delta {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.25;
            margin-top: 0.45rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .sidebar-metric-stack .metric-card {
            min-height: 88px;
        }
        .sidebar-metric-stack .metric-value {
            font-size: 1.45rem;
        }
        div[data-testid="stMetric"] {
            min-height: 104px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        section[data-testid="stSidebar"] div[data-testid="stMetricLabel"],
        section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] * {
            color: var(--muted) !important;
            font-size: 0.8rem;
            font-weight: 800;
            opacity: 1 !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-variant-numeric: tabular-nums;
            font-weight: 900;
        }
        div[data-testid="stMetricDelta"] {
            color: var(--accent) !important;
            font-weight: 800;
            opacity: 1 !important;
        }
        .home-projection-showcase {
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.95fr);
            margin: 0.35rem 0 1rem;
        }
        .favorite-panel,
        .stage-metric-panel,
        .challenger-row {
            background: var(--surface);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            color: var(--ink);
        }
        .favorite-panel {
            background:
                linear-gradient(135deg, rgba(226, 243, 239, 0.72), rgba(255, 255, 255, 0.96)),
                var(--surface);
            display: flex;
            min-height: 210px;
            overflow: hidden;
            padding: 1.35rem 1.45rem;
            position: relative;
        }
        .favorite-panel::before {
            background: var(--accent);
            bottom: 0;
            content: "";
            left: 0;
            position: absolute;
            top: 0;
            width: 5px;
        }
        .favorite-copy {
            align-self: center;
            min-width: 0;
            width: 100%;
        }
        .favorite-copy p,
        .section-kicker {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            margin: 0 0 0.72rem;
            text-transform: uppercase;
        }
        .favorite-title-line {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem 1rem;
        }
        .favorite-title-line strong {
            color: var(--ink);
            font-size: clamp(2rem, 4.6vw, 3.9rem);
            font-weight: 950;
            line-height: 0.98;
            overflow-wrap: anywhere;
            text-transform: uppercase;
        }
        .favorite-title-line span:last-child {
            color: var(--accent);
            font-size: clamp(2.2rem, 5vw, 4.6rem);
            font-variant-numeric: tabular-nums;
            font-weight: 950;
            line-height: 0.94;
            white-space: nowrap;
        }
        .favorite-copy em {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            font-style: normal;
            font-weight: 850;
            letter-spacing: 0.12em;
            margin-top: 0.82rem;
            text-transform: uppercase;
        }
        .favorite-copy small {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.45;
            margin-top: 1.35rem;
        }
        .stage-metric-panel {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            overflow: hidden;
        }
        .stage-metric-tile {
            background: var(--surface);
            border-bottom: 1px solid var(--panel-border);
            border-right: 1px solid var(--panel-border);
            min-height: 104px;
            padding: 1rem;
        }
        .stage-metric-tile:nth-child(2n) {
            border-right: 0;
        }
        .stage-metric-tile:nth-last-child(-n + 2) {
            border-bottom: 0;
        }
        .stage-metric-tile span {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.1em;
            margin-bottom: 0.34rem;
            text-transform: uppercase;
        }
        .stage-metric-tile strong {
            color: var(--ink);
            display: block;
            font-size: clamp(2rem, 3.6vw, 3rem);
            font-variant-numeric: tabular-nums;
            font-weight: 950;
            line-height: 1;
        }
        .challenger-ranking {
            margin: 0.35rem 0 1.35rem;
        }
        .challenger-ranking .section-kicker {
            color: var(--ink-soft);
            margin: 0 0 0.65rem;
        }
        .challenger-list {
            display: grid;
            gap: 0.62rem;
        }
        .challenger-row {
            align-items: center;
            display: grid;
            gap: 0.85rem;
            grid-template-columns: 44px 60px minmax(0, 1fr) auto;
            min-height: 78px;
            padding: 0.78rem 0.95rem;
        }
        .challenger-rank {
            color: var(--muted-soft);
            font-size: 1.45rem;
            font-variant-numeric: tabular-nums;
            font-weight: 950;
            line-height: 1;
        }
        .flag-badge {
            align-items: center;
            background: var(--surface-tint);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-sm);
            display: inline-flex;
            font-size: 1.8rem;
            height: 42px;
            justify-content: center;
            line-height: 1;
            min-width: 54px;
        }
        .flag-fallback {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }
        .challenger-team {
            min-width: 0;
        }
        .challenger-team strong {
            color: var(--ink);
            display: block;
            font-size: 1.08rem;
            font-weight: 900;
            line-height: 1.1;
            overflow: hidden;
            text-overflow: ellipsis;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .challenger-team span {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-top: 0.3rem;
            text-transform: uppercase;
        }
        .challenger-title-prob {
            min-width: 108px;
            text-align: right;
        }
        .challenger-title-prob strong {
            color: var(--accent);
            display: block;
            font-size: 2.15rem;
            font-variant-numeric: tabular-nums;
            font-weight: 950;
            line-height: 0.94;
        }
        .challenger-title-prob span {
            color: var(--muted);
            display: block;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.1em;
            margin-top: 0.25rem;
            text-transform: uppercase;
        }
        .group-grid {
            display: grid;
            gap: 0.85rem;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr));
            margin-top: 1rem;
        }
        .group-card {
            overflow: hidden;
        }
        .group-card-title {
            align-items: center;
            background: linear-gradient(90deg, var(--surface-tint), #ffffff);
            border-bottom: 1px solid var(--panel-border);
            color: var(--ink);
            display: flex;
            gap: 0.55rem;
            padding: 0.75rem 0.88rem;
        }
        .group-card-title span {
            align-items: center;
            background: var(--accent-soft);
            border: 1px solid rgba(8, 120, 108, 0.18);
            border-radius: 999px;
            color: var(--accent-strong);
            display: inline-flex;
            font-weight: 900;
            height: 1.45rem;
            justify-content: center;
            width: 1.45rem;
        }
        .group-card-title strong {
            color: var(--ink);
        }
        .group-table-wrap,
        .projection-table-wrap,
        .prob-table-wrap {
            overflow-x: auto;
            width: 100%;
        }
        .group-card table,
        .projection-table,
        .prob-table {
            border-collapse: collapse;
            width: 100%;
        }
        .group-card .group-table.predicted {
            min-width: 520px;
        }
        .group-card .group-table.actual {
            min-width: 470px;
        }
        .group-card th,
        .group-card td,
        .projection-table th,
        .projection-table td,
        .prob-table th,
        .prob-table td {
            border-bottom: 1px solid #edf1f5;
            color: var(--ink-soft);
            font-size: 0.78rem;
            padding: 0.52rem 0.48rem;
            text-align: right;
            white-space: nowrap;
        }
        .group-card th,
        .projection-table th,
        .prob-table th {
            background: #f3f7fa;
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 900;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .group-card tr:last-child td,
        .projection-table tr:last-child td,
        .prob-table tr:last-child td {
            border-bottom: 0;
        }
        .group-card tbody tr:hover td,
        .projection-table tbody tr:hover td,
        .prob-table tbody tr:hover td:not(.bucket-cell) {
            background: #f8fbfd;
        }
        .group-card .team-name,
        .projection-table .team-sticky {
            color: var(--ink);
            font-weight: 800;
            max-width: 180px;
            min-width: 140px;
            overflow: hidden;
            text-align: left;
            text-overflow: ellipsis;
        }
        .group-card .rank {
            border-radius: 999px;
            display: inline-block;
            font-variant-numeric: tabular-nums;
            font-weight: 900;
            min-width: 1.45rem;
            padding: 0.12rem 0.26rem;
            text-align: center;
        }
        .group-card .qualified {
            background: var(--accent-soft);
            color: var(--accent-strong);
        }
        .group-card .third {
            background: var(--accent-amber-soft);
            color: #8a5a04;
        }
        .group-card .out {
            background: var(--accent-coral-soft);
            color: #9e3328;
        }
        .group-card .points {
            color: var(--ink);
            font-weight: 950;
        }
        .group-legend {
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 0.85rem 0 1.25rem;
        }
        .dot {
            border-radius: 999px;
            display: inline-block;
            height: 0.62rem;
            margin-right: 0.35rem;
            width: 0.62rem;
        }
        .dot.qualified {
            background: var(--accent);
        }
        .dot.third {
            background: var(--accent-amber);
        }
        .dot.out {
            background: var(--accent-coral);
        }
        .projection-table-wrap,
        .bracket-board {
            margin-top: 1rem;
        }
        .projection-table {
            min-width: 1180px;
        }
        .projection-table .team-sticky {
            background: inherit;
            left: 0;
            position: sticky;
            z-index: 1;
        }
        .projection-table thead .team-sticky {
            background: #f3f7fa;
            z-index: 2;
        }
        .team-code {
            color: var(--muted);
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 900;
            margin-right: 0.45rem;
            min-width: 2.2rem;
        }
        .bracket-board {
            align-items: stretch;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(6, minmax(210px, 1fr));
            overflow-x: auto;
            padding: 1rem;
        }
        .bracket-column h4 {
            color: var(--ink);
            font-size: 0.9rem;
            margin: 0 0 0.75rem;
            text-align: center;
        }
        .bracket-stack {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
        }
        .bracket-card {
            background: var(--surface-raised);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            padding: 0.68rem;
        }
        .final-card {
            background: var(--accent-amber-soft);
            border-color: rgba(199, 132, 20, 0.45);
            box-shadow: 0 0 0 2px rgba(199, 132, 20, 0.12);
        }
        .match-meta,
        .winner-line {
            color: var(--muted);
            font-size: 0.72rem;
        }
        .match-meta {
            display: flex;
            gap: 0.5rem;
            justify-content: space-between;
        }
        .match-teams {
            display: grid;
            gap: 0.25rem;
            margin: 0.45rem 0;
        }
        .match-teams span,
        .winner-line strong {
            color: var(--ink);
            font-weight: 850;
            overflow-wrap: anywhere;
        }
        .match-teams strong {
            color: var(--accent);
            font-variant-numeric: tabular-nums;
        }
        .match-prediction-grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
            margin-top: 1rem;
        }
        .match-prediction-card {
            min-width: 0;
            padding: 1rem;
        }
        .match-prediction-topline {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 0.74rem;
            font-weight: 850;
            gap: 0.75rem;
            justify-content: space-between;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
        }
        .source-pill,
        .prediction-status {
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 900;
            line-height: 1;
            padding: 0.36rem 0.52rem;
            white-space: nowrap;
        }
        .source-pill {
            background: var(--accent-blue-soft);
            color: var(--accent-blue);
        }
        .match-scoreboard {
            align-items: center;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        }
        .match-team {
            display: grid;
            gap: 0.25rem;
            min-width: 0;
        }
        .match-team-away {
            text-align: right;
        }
        .match-team .team-code {
            color: var(--muted);
            font-size: 0.72rem;
        }
        .match-team strong {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 900;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .score-block {
            background: var(--ink);
            border-radius: var(--radius);
            color: #ffffff;
            min-width: 92px;
            padding: 0.65rem 0.8rem;
            text-align: center;
        }
        .score-block.is-pending {
            background: var(--surface-raised);
            border: 1px solid var(--panel-border);
            color: var(--ink);
        }
        .score-block strong {
            display: block;
            font-size: 1.55rem;
            font-variant-numeric: tabular-nums;
            font-weight: 950;
            line-height: 1;
        }
        .score-block span {
            color: inherit;
            display: block;
            font-size: 0.66rem;
            font-weight: 850;
            margin-top: 0.35rem;
            opacity: 0.78;
            text-transform: uppercase;
        }
        .prediction-summary {
            align-items: center;
            background: var(--surface-tint);
            border: 1px solid #e4ebf1;
            border-radius: var(--radius);
            display: grid;
            gap: 0.25rem 0.65rem;
            grid-template-columns: 1fr auto;
            margin: 1rem 0 0.75rem;
            padding: 0.75rem;
        }
        .prediction-summary span {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 850;
            grid-column: 1 / -1;
            text-transform: uppercase;
        }
        .prediction-summary strong {
            color: var(--ink);
            font-weight: 900;
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .prediction-summary em {
            color: var(--accent);
            font-style: normal;
            font-variant-numeric: tabular-nums;
            font-weight: 950;
            white-space: nowrap;
        }
        .probability-visual {
            display: grid;
            gap: 0.55rem;
        }
        .probability-bar {
            background: #e8eef4;
            border-radius: 999px;
            display: flex;
            height: 0.85rem;
            overflow: hidden;
            width: 100%;
        }
        .probability-segment.home {
            background: var(--accent);
        }
        .probability-segment.draw {
            background: var(--accent-blue);
        }
        .probability-segment.away {
            background: var(--accent-coral);
        }
        .probability-segment.is-favorite {
            box-shadow: inset 0 0 0 999px rgba(255, 255, 255, 0.12);
        }
        .probability-legend {
            display: grid;
            gap: 0.35rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .probability-legend span {
            min-width: 0;
        }
        .probability-legend strong,
        .probability-legend em {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .probability-legend strong {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
        }
        .probability-legend em {
            color: var(--ink);
            font-size: 0.9rem;
            font-style: normal;
            font-variant-numeric: tabular-nums;
            font-weight: 900;
        }
        .probability-legend .is-favorite strong,
        .probability-legend .is-favorite em {
            color: var(--accent);
        }
        .prediction-details {
            border-top: 1px solid #edf1f5;
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.9rem 0 0;
            padding-top: 0.75rem;
        }
        .prediction-details div {
            min-width: 0;
        }
        .prediction-details dt {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 850;
            margin: 0 0 0.25rem;
            text-transform: uppercase;
        }
        .prediction-details dd {
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 900;
            margin: 0;
            overflow-wrap: anywhere;
        }
        .prediction-status.hit {
            background: var(--accent-soft);
            color: var(--accent-strong);
        }
        .prediction-status.miss {
            background: var(--accent-coral-soft);
            color: #9e3328;
        }
        .prediction-status.pending {
            background: #eef2f6;
            color: var(--muted);
        }
        .prob-table-wrap {
            margin-top: 1rem;
        }
        .prob-table {
            font-size: 15px;
            line-height: 1.25;
            min-width: 1180px;
        }
        .prob-table th {
            text-align: center;
        }
        .prob-table th span {
            display: inline-block;
        }
        .prob-table td {
            background: var(--surface);
            padding: 0.62rem 0.75rem;
        }
        .prob-table tr.even-row td:not(.bucket-cell) {
            background: #f8fbfd;
        }
        .prob-table .bucket-cell {
            background: var(--accent);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 900;
            min-width: 72px;
            text-align: center;
            vertical-align: middle;
            width: 72px;
        }
        .prob-table .team-cell {
            color: var(--ink);
            font-weight: 750;
            min-width: 190px;
            text-align: left;
        }
        .prob-table .numeric-cell {
            font-variant-numeric: tabular-nums;
            min-width: 120px;
            text-align: center;
        }
        .prob-table .score-cell {
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            font-weight: 900;
            min-width: 110px;
            text-align: center;
        }
        .prob-table .actual-score {
            color: var(--ink-soft);
        }
        .prob-table .result-cell {
            font-weight: 900;
            min-width: 118px;
            text-align: center;
        }
        .prob-table .source-cell {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 850;
            min-width: 72px;
            text-align: center;
        }
        .prob-table .result-hit {
            color: var(--accent-strong);
        }
        .prob-table .result-miss {
            color: #9e3328;
        }
        .prob-table .result-pending {
            color: var(--muted-soft);
            font-weight: 700;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border-radius: var(--radius);
            overflow: hidden;
        }
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
            gap: 0.7rem;
        }
        @media (prefers-reduced-motion: reduce) {
            html {
                scroll-behavior: auto;
            }
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }
        @media (max-width: 1100px) {
            .home-projection-showcase {
                grid-template-columns: 1fr;
            }
            .metric-grid,
            .metric-grid-four {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .metric-grid-three {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .group-grid {
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr));
            }
            .bracket-board {
                grid-template-columns: repeat(3, minmax(210px, 1fr));
            }
        }
        @media (max-width: 760px) {
            .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }
            .dashboard-hero {
                margin-top: 2.75rem;
                padding: 1.05rem 1.1rem;
            }
            .metric-grid,
            .metric-grid-four,
            .metric-grid-three {
                grid-template-columns: 1fr;
            }
            .group-grid {
                grid-template-columns: 1fr;
            }
            .bracket-board {
                grid-template-columns: repeat(2, minmax(210px, 1fr));
            }
            .favorite-panel {
                min-height: 0;
            }
            .challenger-row {
                grid-template-columns: 38px 52px minmax(0, 1fr);
            }
            .challenger-title-prob {
                grid-column: 3;
                min-width: 0;
                text-align: left;
            }
            .challenger-title-prob strong {
                font-size: 1.9rem;
            }
            .match-scoreboard {
                grid-template-columns: 1fr;
                text-align: left;
            }
            .match-team-away {
                text-align: left;
            }
            .score-block {
                justify-self: stretch;
            }
            .prediction-details,
            .probability-legend {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
