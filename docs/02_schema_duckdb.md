# Schema DuckDB

- Objetivo: manter tabelas analíticas locais e reproduzíveis.
- Fonte primária: arquivos brutos em `data/raw`.
- Camadas sugeridas: `raw`, `staging`, `analytics`.
- Fontes de força das seleções: `f_elo_history` guarda o World Cup Probability
  Elo calculado pelo projeto; `d_world_football_elo_ratings` guarda o World
  Football Elo Ratings; `d_fifa_world_ranking` guarda o FIFA World Ranking com
  datas oficiais de atualização.
- Histórico específico de Copa: `d_world_cup_prior_team_history` guarda
  participações, pontos por jogo e saldo de gols por jogo em Copas masculinas
  anteriores ao ano da observação, a partir da Fjelstul World Cup Database.
- Disciplina em Copa: `d_world_cup_prior_discipline_history` guarda cartões
  amarelos por jogo, expulsões por jogo e penalidade de fair play por jogo em
  Copas masculinas anteriores ao ano da observação, também a partir da Fjelstul.
