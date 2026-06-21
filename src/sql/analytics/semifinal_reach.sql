CREATE OR REPLACE VIEW v_semifinal_reach AS
WITH semifinal_participants AS (
    SELECT
        simulation_id,
        home_team_id AS team_id,
        home_team_name AS team_name
    FROM simulated_results
    WHERE round_name = 'semifinal'
    UNION ALL
    SELECT
        simulation_id,
        away_team_id AS team_id,
        away_team_name AS team_name
    FROM simulated_results
    WHERE round_name = 'semifinal'
),

total_sims AS (
    SELECT COUNT(DISTINCT simulation_id) AS total_simulations
    FROM simulated_results
)

SELECT
    sp.team_id,
    sp.team_name,
    ROUND(
        100.0 * COUNT(DISTINCT sp.simulation_id) / NULLIF(ts.total_simulations, 0),
        2
    ) AS semifinal_pct,
    COUNT(DISTINCT sp.simulation_id) AS semifinal_appearances,
    ts.total_simulations
FROM semifinal_participants AS sp
CROSS JOIN total_sims AS ts
GROUP BY sp.team_id, sp.team_name, ts.total_simulations
ORDER BY semifinal_pct DESC, sp.team_name ASC
