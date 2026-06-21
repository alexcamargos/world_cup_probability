CREATE OR REPLACE VIEW v_title_probability AS
WITH champions AS (
    SELECT
        simulation_id,
        winner_team_id AS team_id,
        CASE
            WHEN winner_team_id = home_team_id THEN home_team_name
            ELSE away_team_name
        END AS team_name
    FROM simulated_results
    WHERE round_name = 'final'
),

total_sims AS (
    SELECT COUNT(DISTINCT simulation_id) AS total_simulations
    FROM simulated_results
)

SELECT
    c.team_id,
    c.team_name,
    ROUND(100.0 * COUNT(*) / NULLIF(ts.total_simulations, 0), 2) AS title_probability,
    COUNT(*) AS title_wins,
    ts.total_simulations
FROM champions AS c
CROSS JOIN total_sims AS ts
GROUP BY c.team_id, c.team_name, ts.total_simulations
ORDER BY title_probability DESC, c.team_name ASC
