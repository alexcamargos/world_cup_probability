# Schema DuckDB

- Objetivo: manter tabelas analíticas locais e reproduzíveis.
- Fonte primária: arquivos brutos em `data/raw`.
- Camadas sugeridas: `raw`, `staging`, `analytics`.
- Fontes de força das seleções: `f_elo_history` guarda o World Cup Probability
  Elo calculado pelo projeto; `d_world_football_elo_ratings` guarda o World
  Football Elo Ratings; `d_fifa_world_ranking` guarda o FIFA World Ranking com
  datas oficiais de atualização.
