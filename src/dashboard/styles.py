"""CSS injection for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --dashboard-bg: #f6f7f9;
            --panel-border: #d8dde6;
            --ink: #18202f;
            --muted: #667085;
            --accent: #0f6b5f;
            --accent-soft: #e6f4f1;
        }
        .stApp {
            background: var(--dashboard-bg);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        .dashboard-hero {
            background:
                linear-gradient(135deg, rgba(15, 107, 95, 0.94), rgba(24, 32, 47, 0.96)),
                radial-gradient(circle at 78% 20%, rgba(255, 255, 255, 0.22), transparent 32%);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            color: #ffffff;
            margin-bottom: 1.25rem;
            padding: 1.35rem 1.5rem;
        }
        .dashboard-hero .eyebrow {
            color: #c8f1e8;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            margin: 0 0 0.4rem;
            text-transform: uppercase;
        }
        .dashboard-hero h1 {
            color: #ffffff;
            font-size: 2rem;
            line-height: 1.1;
            margin: 0;
            letter-spacing: 0;
        }
        .dashboard-hero p:last-child {
            color: rgba(255, 255, 255, 0.84);
            margin: 0.65rem 0 0;
            max-width: 760px;
        }
        .home-projection-showcase {
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.95fr);
            margin: 0.35rem 0 1rem;
        }
        .favorite-panel,
        .stage-metric-panel,
        .challenger-row {
            background:
                linear-gradient(115deg, rgba(20, 91, 63, 0.92), rgba(8, 34, 29, 0.98)),
                repeating-linear-gradient(
                    90deg,
                    rgba(255, 255, 255, 0.035) 0,
                    rgba(255, 255, 255, 0.035) 1px,
                    transparent 1px,
                    transparent 52px
                );
            border: 1px solid rgba(111, 163, 135, 0.45);
            color: #fbf6e8;
        }
        .favorite-panel {
            border-left: 3px solid #f6c445;
            border-radius: 8px;
            display: flex;
            min-height: 220px;
            padding: 1.45rem 1.6rem;
        }
        .favorite-copy {
            align-self: center;
            min-width: 0;
            width: 100%;
        }
        .favorite-copy p,
        .section-kicker {
            color: #f6c445;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.18em;
            margin: 0 0 0.8rem;
            text-transform: uppercase;
        }
        .favorite-title-line {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem 1rem;
        }
        .favorite-title-line strong {
            color: #fff8e6;
            font-size: clamp(2.2rem, 5vw, 4.25rem);
            font-weight: 950;
            line-height: 0.95;
            overflow-wrap: anywhere;
            text-transform: uppercase;
        }
        .favorite-title-line span:last-child {
            color: #f6c445;
            font-size: clamp(2.4rem, 6vw, 5.1rem);
            font-weight: 950;
            line-height: 0.9;
            white-space: nowrap;
        }
        .favorite-copy em {
            color: rgba(251, 246, 232, 0.76);
            display: block;
            font-size: 0.82rem;
            font-style: normal;
            font-weight: 900;
            letter-spacing: 0.16em;
            margin-top: 0.85rem;
            text-transform: uppercase;
        }
        .favorite-copy small {
            color: rgba(251, 246, 232, 0.56);
            display: block;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-top: 1.45rem;
            text-transform: uppercase;
        }
        .stage-metric-panel {
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            overflow: hidden;
        }
        .stage-metric-tile {
            border-bottom: 1px solid rgba(111, 163, 135, 0.28);
            border-right: 1px solid rgba(111, 163, 135, 0.28);
            min-height: 110px;
            padding: 1.05rem;
        }
        .stage-metric-tile:nth-child(2n) {
            border-right: 0;
        }
        .stage-metric-tile:nth-last-child(-n + 2) {
            border-bottom: 0;
        }
        .stage-metric-tile span {
            color: rgba(251, 246, 232, 0.62);
            display: block;
            font-size: 0.75rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }
        .stage-metric-tile strong {
            color: #fbf6e8;
            display: block;
            font-size: clamp(2.2rem, 4vw, 3.2rem);
            font-weight: 950;
            line-height: 1;
        }
        .challenger-ranking {
            margin: 0.3rem 0 1.35rem;
        }
        .challenger-ranking .section-kicker {
            color: #344054;
            margin: 0 0 0.65rem;
        }
        .challenger-list {
            display: grid;
            gap: 0.55rem;
        }
        .challenger-row {
            align-items: center;
            border-radius: 0;
            display: grid;
            gap: 0.9rem;
            grid-template-columns: 44px 60px minmax(0, 1fr) auto;
            min-height: 86px;
            padding: 0.85rem 1rem;
        }
        .challenger-rank {
            color: rgba(251, 246, 232, 0.52);
            font-size: 1.65rem;
            font-weight: 950;
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }
        .flag-badge {
            align-items: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            display: inline-flex;
            font-size: 2.05rem;
            height: 42px;
            justify-content: center;
            line-height: 1;
            min-width: 54px;
        }
        .flag-fallback {
            color: #fbf6e8;
            font-size: 0.85rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }
        .challenger-team {
            min-width: 0;
        }
        .challenger-team strong {
            color: #fff8e6;
            display: block;
            font-size: 1.25rem;
            font-weight: 950;
            line-height: 1.1;
            overflow: hidden;
            text-overflow: ellipsis;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .challenger-team span {
            color: rgba(251, 246, 232, 0.55);
            display: block;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-top: 0.28rem;
            text-transform: uppercase;
        }
        .challenger-title-prob {
            min-width: 108px;
            text-align: right;
        }
        .challenger-title-prob strong {
            color: #fff8e6;
            display: block;
            font-size: 2.65rem;
            font-weight: 950;
            font-variant-numeric: tabular-nums;
            line-height: 0.9;
        }
        .challenger-title-prob span {
            color: rgba(251, 246, 232, 0.55);
            display: block;
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            margin-top: 0.25rem;
            text-transform: uppercase;
        }
        .metric-grid {
            display: grid;
            gap: 0.85rem;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            margin: 1rem 0 1.25rem;
        }
        .metric-grid-four {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .metric-grid-three {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .sidebar-metric-stack {
            display: grid;
            gap: 0.8rem;
            margin-top: 1rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-width: 0;
            padding: 0.85rem 1rem;
        }
        .metric-label {
            color: #344054;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.55rem;
            opacity: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .metric-value {
            color: #101828;
            font-size: 1.75rem;
            font-weight: 900;
            line-height: 1.1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .metric-delta {
            color: #067647;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.45rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .sidebar-metric-stack .metric-card {
            min-height: 96px;
        }
        .sidebar-metric-stack .metric-value {
            font-size: 1.65rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            min-height: 108px;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        section[data-testid="stSidebar"] div[data-testid="stMetricLabel"],
        section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] * {
            color: #344054 !important;
            font-size: 0.82rem;
            font-weight: 700;
            opacity: 1 !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-weight: 800;
        }
        div[data-testid="stMetricDelta"] {
            color: #067647 !important;
            font-weight: 700;
            opacity: 1 !important;
        }
        div[data-testid="stTabs"] button {
            color: #344054;
            font-weight: 700;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f6b5f;
        }
        input, textarea, select {
            color: #101828 !important;
            background: #ffffff !important;
        }
        button:focus-visible,
        [role="tab"]:focus-visible,
        input:focus-visible {
            outline: 3px solid rgba(15, 107, 95, 0.35) !important;
            outline-offset: 2px;
        }
        h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        p, label, span {
            text-wrap: pretty;
        }
        .group-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .group-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            overflow: hidden;
        }
        .group-table-wrap {
            overflow-x: auto;
            width: 100%;
        }
        .group-card-title {
            align-items: center;
            background: #18202f;
            color: #ffffff;
            display: flex;
            gap: 0.55rem;
            padding: 0.7rem 0.85rem;
        }
        .group-card-title span {
            align-items: center;
            background: #ffffff;
            border-radius: 999px;
            color: #18202f;
            display: inline-flex;
            font-weight: 800;
            height: 1.45rem;
            justify-content: center;
            width: 1.45rem;
        }
        .group-card table,
        .projection-table {
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
        .projection-table td {
            border-bottom: 1px solid #edf0f5;
            color: #344054;
            font-size: 0.78rem;
            padding: 0.48rem 0.45rem;
            text-align: right;
            white-space: nowrap;
        }
        .group-card th,
        .projection-table th {
            background: #f8fafc;
            color: #667085;
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .group-card .team-name,
        .projection-table .team-sticky {
            font-weight: 700;
            max-width: 180px;
            min-width: 140px;
            overflow: hidden;
            text-align: left;
            text-overflow: ellipsis;
        }
        .group-card .rank {
            border-radius: 999px;
            display: inline-block;
            font-weight: 800;
            min-width: 1.45rem;
            padding: 0.1rem 0.25rem;
            text-align: center;
        }
        .group-card .qualified {
            background: #e6f4f1;
            color: #0f6b5f;
        }
        .group-card .third {
            background: #fff4df;
            color: #9a5b00;
        }
        .group-card .out {
            background: #fee4e2;
            color: #b42318;
        }
        .group-card .points {
            color: var(--ink);
            font-weight: 900;
        }
        .group-legend {
            color: #475467;
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
            background: #0f6b5f;
        }
        .dot.third {
            background: #d68b00;
        }
        .dot.out {
            background: #d65f4c;
        }
        .projection-table-wrap,
        .bracket-board {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            margin-top: 1rem;
            overflow-x: auto;
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
            background: #f8fafc;
            z-index: 2;
        }
        .team-code {
            color: #667085;
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 800;
            margin-right: 0.45rem;
            min-width: 2.2rem;
        }
        .bracket-board {
            align-items: stretch;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(6, minmax(210px, 1fr));
            padding: 1rem;
        }
        .bracket-column h4 {
            color: #18202f;
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
            background: #f8fafc;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            padding: 0.65rem;
        }
        .final-card {
            background: #fff8e8;
            border-color: #d68b00;
            box-shadow: 0 0 0 2px rgba(214, 139, 0, 0.14);
        }
        .match-meta,
        .winner-line {
            color: #667085;
            font-size: 0.72rem;
        }
        .match-meta {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
        }
        .match-teams {
            display: grid;
            gap: 0.25rem;
            margin: 0.45rem 0;
        }
        .match-teams span,
        .winner-line strong {
            color: #18202f;
            font-weight: 800;
            overflow-wrap: anywhere;
        }
        .match-teams strong {
            color: #0f6b5f;
            font-variant-numeric: tabular-nums;
        }
        .match-prediction-grid {
            display: grid;
            gap: 0.85rem;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
            margin-top: 1rem;
        }
        .match-prediction-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-width: 0;
            padding: 1rem;
        }
        .match-prediction-topline {
            align-items: center;
            color: #667085;
            display: flex;
            font-size: 0.74rem;
            font-weight: 800;
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
            padding: 0.35rem 0.5rem;
            white-space: nowrap;
        }
        .source-pill {
            background: #e6f4f1;
            color: #0f6b5f;
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
            color: #667085;
            font-size: 0.72rem;
            font-weight: 900;
        }
        .match-team strong {
            color: #18202f;
            font-size: 1.1rem;
            font-weight: 900;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .score-block {
            background: #18202f;
            border-radius: 8px;
            color: #ffffff;
            min-width: 92px;
            padding: 0.65rem 0.8rem;
            text-align: center;
        }
        .score-block.is-pending {
            background: #f8fafc;
            border: 1px solid #d8dde6;
            color: #18202f;
        }
        .score-block strong {
            display: block;
            font-size: 1.55rem;
            font-variant-numeric: tabular-nums;
            font-weight: 900;
            line-height: 1;
        }
        .score-block span {
            color: inherit;
            display: block;
            font-size: 0.66rem;
            font-weight: 800;
            margin-top: 0.35rem;
            opacity: 0.78;
            text-transform: uppercase;
        }
        .prediction-summary {
            align-items: center;
            background: #f8fafc;
            border: 1px solid #edf0f5;
            border-radius: 8px;
            display: grid;
            gap: 0.25rem 0.65rem;
            grid-template-columns: 1fr auto;
            margin: 1rem 0 0.75rem;
            padding: 0.75rem;
        }
        .prediction-summary span {
            color: #667085;
            font-size: 0.72rem;
            font-weight: 800;
            grid-column: 1 / -1;
            text-transform: uppercase;
        }
        .prediction-summary strong {
            color: #18202f;
            font-weight: 900;
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .prediction-summary em {
            color: #0f6b5f;
            font-style: normal;
            font-weight: 900;
            white-space: nowrap;
        }
        .probability-visual {
            display: grid;
            gap: 0.55rem;
        }
        .probability-bar {
            background: #edf0f5;
            border-radius: 999px;
            display: flex;
            height: 0.85rem;
            overflow: hidden;
            width: 100%;
        }
        .probability-segment.home {
            background: #0f6b5f;
        }
        .probability-segment.draw {
            background: #9aa4b2;
        }
        .probability-segment.away {
            background: #d65f4c;
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
            color: #475467;
            font-size: 0.72rem;
            font-weight: 800;
        }
        .probability-legend em {
            color: #18202f;
            font-size: 0.9rem;
            font-style: normal;
            font-weight: 900;
            font-variant-numeric: tabular-nums;
        }
        .probability-legend .is-favorite strong,
        .probability-legend .is-favorite em {
            color: #0f6b5f;
        }
        .prediction-details {
            border-top: 1px solid #edf0f5;
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
            color: #667085;
            font-size: 0.7rem;
            font-weight: 800;
            margin: 0 0 0.25rem;
            text-transform: uppercase;
        }
        .prediction-details dd {
            color: #18202f;
            font-size: 0.88rem;
            font-weight: 900;
            margin: 0;
            overflow-wrap: anywhere;
        }
        .prediction-status.hit {
            background: #e6f4f1;
            color: #067647;
        }
        .prediction-status.miss {
            background: #fee4e2;
            color: #b42318;
        }
        .prediction-status.pending {
            background: #f1f3f7;
            color: #667085;
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
                font-size: 2rem;
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
        .prob-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin-top: 1rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #ffffff;
        }
        .prob-table {
            width: 100%;
            min-width: 1180px;
            border-collapse: collapse;
            font-size: 15px;
            line-height: 1.25;
        }
        .prob-table th {
            background: #18202f;
            color: #ffffff;
            border-bottom: 1px solid #2f3949;
            padding: 10px 12px;
            text-align: center;
            font-weight: 700;
            letter-spacing: 0;
        }
        .prob-table th span {
            display: inline-block;
        }
        .prob-table td {
            border-bottom: 1px solid #edf0f5;
            padding: 10px 12px;
            color: #344054;
            background: #ffffff;
        }
        .prob-table tr.even-row td:not(.bucket-cell) {
            background: #f8fafc;
        }
        .prob-table .bucket-cell {
            width: 72px;
            min-width: 72px;
            text-align: center;
            vertical-align: middle;
            background: var(--accent);
            color: #ffffff;
            font-weight: 800;
            font-size: 18px;
        }
        .prob-table .team-cell {
            min-width: 190px;
            font-weight: 500;
        }
        .prob-table .numeric-cell {
            min-width: 120px;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }
        .prob-table .score-cell {
            min-width: 110px;
            text-align: center;
            color: var(--ink);
            font-weight: 800;
            font-variant-numeric: tabular-nums;
        }
        .prob-table .actual-score {
            color: #38445c;
        }
        .prob-table .result-cell {
            min-width: 118px;
            text-align: center;
            font-weight: 800;
        }
        .prob-table .source-cell {
            color: #475467;
            font-size: 0.82rem;
            font-weight: 800;
            min-width: 72px;
            text-align: center;
        }
        .prob-table .result-hit {
            color: #067647;
        }
        .prob-table .result-miss {
            color: #b42318;
        }
        .prob-table .result-pending {
            color: #8993a5;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
