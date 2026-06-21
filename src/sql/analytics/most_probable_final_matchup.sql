CREATE OR REPLACE VIEW v_most_probable_final_matchup AS
WITH finals AS (
    SELECT
        simulation_id,
        CASE
            WHEN home_team_id <= away_team_id THEN home_team_id
            ELSE away_team_id
        END AS team_a_id,
        CASE
            WHEN home_team_id <= away_team_id THEN home_team_name
            ELSE away_team_name
        END AS team_a_name,
        CASE
            WHEN home_team_id <= away_team_id THEN away_team_id
            ELSE home_team_id
        END AS team_b_id,
        CASE
            WHEN home_team_id <= away_team_id THEN away_team_name
            ELSE home_team_name
        END AS team_b_name
    FROM simulated_results
    WHERE round_name = 'final'
),

pair_counts AS (
    SELECT
        team_a_id,
        team_b_id,
        team_a_name,
        team_b_name,
        COUNT(*) AS matchup_count
    FROM finals
    GROUP BY team_a_id, team_b_id, team_a_name, team_b_name
),

total_sims AS (
    SELECT COUNT(DISTINCT simulation_id) AS total_simulations
    FROM simulated_results
)

SELECT
    matchup_id,
    matchup_label,
    matchup_probability,
    matchup_count,
    total_simulations
FROM (
    SELECT
        CONCAT(pc.team_a_id, '_vs_', pc.team_b_id) AS matchup_id,
        CONCAT(pc.team_a_name, ' vs ', pc.team_b_name) AS matchup_label,
        ROUND(
            100.0 * pc.matchup_count / NULLIF(ts.total_simulations, 0),
            2
        ) AS matchup_probability,
        pc.matchup_count,
        ts.total_simulations,
        ROW_NUMBER() OVER (
            ORDER BY pc.matchup_count DESC, pc.team_a_name ASC, pc.team_b_name ASC
        ) AS rn
    FROM pair_counts AS pc
    CROSS JOIN total_sims AS ts
) AS ranked
WHERE rn = 1
