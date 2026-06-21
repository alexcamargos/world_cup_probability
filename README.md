# World Cup Probability

Base do repositório para modelagem de probabilidades da Copa do Mundo.

## Execução do pipeline

Ordem recomendada:

1. Baixar os dados brutos para `data/raw`:
   - `uv run download-data`
   - `uv run download-data --sources matches`
   - `uv run download-data --sources matches squad --ea-fc-dataset flynn28/eafc26-player-database`
   - `uv run download-data --sources fbref --fbref-leagues <league> --fbref-seasons <season>`
   - `uv run download-data --sources transfermarkt --transfermarkt-manifest data/raw/transfermarkt/teams.json`
   - Para baixar todas as fontes configuradas numa chamada:
     `uv run download-data --ea-fc-dataset flynn28/eafc26-player-database --fbref-leagues <league> --fbref-seasons <season> --transfermarkt-manifest data/raw/transfermarkt/teams.json`
2. Carregar os dados brutos no warehouse DuckDB:
   - `uv run load-data`
   - `uv run load-data --sources matches`
   - `uv run load-data --sources matches squad --ea-fc-dataset flynn28/eafc26-player-database`
   - `uv run load-data --sources fbref`
   - `uv run load-data --sources transfermarkt`
   - Para carregar todas as fontes já baixadas numa chamada:
     `uv run load-data --ea-fc-dataset flynn28/eafc26-player-database`
3. Coletar e carregar em um único comando, quando for conveniente:
   - `uv run collection --sources matches`
   - `uv run collection --sources matches --load-existing`
4. Inicializar o warehouse e carregar dados brutos:
   - `uv run db-init`
5. Calcular o histórico World Cup Probability Elo:
   - `uv run world-cup-probability-elo`
6. Carregar o World Football Elo Ratings:
   - `uv run world-football-elo-ratings`
   - Para recarregar um snapshot local sem rede:
     `uv run world-football-elo-ratings --load-existing --raw-path data/raw/eloratings/world_football_elo_ratings_snapshot.jsonl`
7. Carregar o FIFA World Ranking:
   - `uv run fifa-world-ranking`
   - Para recarregar um snapshot local sem rede:
     `uv run fifa-world-ranking --load-existing --raw-path data/raw/fifa_world_ranking/men_snapshot.jsonl`
8. Gerar a base de features:
   - `uv run features`
9. Treinar o modelo Poisson e gerar SHAP:
   - `uv run train-model`
10. Rodar a simulação Monte Carlo:
   - `uv run simulate`
   - Exemplo reproduzível e menor para validação local:
     `uv run simulate --iterations 1000 --batch-size 250 --seed 42`
   - Para exportar os CSVs analíticos na mesma execução:
     `uv run simulate --iterations 100000 --batch-size 2500 --export-analytics`
11. Gerar as análises e CSVs:
   - `uv run analytics`
12. Orquestrar tudo em sequência:
   - `uv run pipeline --iterations 100000 --batch-size 2500`
13. Abrir a interface web Streamlit:
   - `uv run dashboard`
   - Alternativamente: `uv run streamlit run src/app.py`

A interface web lê `data/warehouse/world_cup.duckdb`, permite selecionar a
rodada da Copa e apresenta as probabilidades agregadas de vitória do Time 1,
empate e vitória do Time 2 em uma tabela no formato do relatório. Para fases
eliminatórias, os confrontos podem variar entre simulações; por padrão, a tela
mostra o confronto mais frequente de cada jogo, com opção para exibir todos.

Por regra de negócio, a coleta histórica carrega apenas partidas em ou após
`2010-01-01`. O valor pode ser alterado manualmente com `--cutoff-date`.

O valor de `--ea-fc-dataset` deve ser um slug real do Kaggle, no formato
`owner/dataset`, copiado da URL depois de `/datasets/`. Exemplos de datasets
compatíveis para atributos de jogadores são `flynn28/eafc26-player-database`,
`aniss7/fifa-player-data-from-sofifa-2025-06-03` e
`rehandl23/fifa-24-player-stats-dataset`. A disponibilidade depende da licença
e visibilidade do dataset na sua conta Kaggle.

Sem argumentos, `download-data` usa os defaults do projeto: baixa partidas
históricas do Kaggle, baixa o dataset padrão de atributos EA FC/FIFA e tenta
Transfermarkt apenas se houver `config/transfermarkt_teams.json`. FBref fica
como fonte opcional explícita porque o site costuma bloquear coleta automatizada
com HTTP 403.

O comando `world-football-elo-ratings` baixa o snapshot global atual de
[World Football Elo Ratings](https://www.eloratings.net/) a partir dos TSVs
públicos do site (`World.tsv` e `en.teams.tsv`) e persiste as tabelas
`d_world_football_elo_ratings` e `d_world_football_elo_team_aliases`. A feature
`world_football_elo_ratings_diff` entra no treino junto com
`world_cup_probability_elo_diff`; se o snapshot ainda não estiver carregado, a
geração de features usa o World Cup Probability Elo como fallback para manter o
pipeline executável.

O comando `fifa-world-ranking` baixa o snapshot masculino atual do
[FIFA World Ranking](https://inside.fifa.com/fifa-world-ranking/men), persiste
as tabelas `d_fifa_world_ranking` e `d_fifa_world_ranking_team_aliases`, e
mantém a data da última atualização oficial e da próxima atualização oficial no
warehouse. Na página oficial consultada, a última atualização é 11 de junho de
2026 e a próxima é 20 de julho de 2026. As features
`fifa_world_ranking_points_diff` e `fifa_world_ranking_rank_diff` entram no
treino junto com `world_cup_probability_elo_diff` e
`world_football_elo_ratings_diff`; se o snapshot ainda não estiver carregado,
a geração de features usa o World Cup Probability Elo como fallback para pontos
e rank neutro para manter o pipeline executável.

O comando `simulate` carrega `models/xgb_poisson_model.json`, lê o warehouse
`data/warehouse/world_cup.duckdb`, monta as intensidades de gols das 48
seleções oficiais da Copa de 2026 a partir das features do projeto e executa a
tabela real do torneio: 12 grupos de 4, 72 jogos de fase de grupos, melhores
oito terceiros colocados, fase de 32 avos, oitavas, quartas, semifinais, disputa
de terceiro lugar e final. A agenda real é o alvo da simulação; placares reais
da Copa de 2026 não são usados no treino, no Elo, nas features, na forma recente
ou como resultado preservado pelo simulador.

Os manifestos padrão ficam em:

- `config/fbref_sources.json`
- `config/transfermarkt_teams.json`

No manifesto do Transfermarkt, `url` é opcional. Quando ela não é informada, o
download tenta resolver a página da seleção pela busca do Transfermarkt usando
`team_name` ou `search_query`.

Manifesto Transfermarkt esperado:

```json
{
  "teams": [
    {
      "team_id": "Brazil",
      "team_name": "Brazil",
      "url": "https://www.transfermarkt.com/..."
    }
  ]
}
```

## Saídas principais

- `data/warehouse/world_cup.duckdb`: fonte analítica local
- `models/xgb_poisson_model.json`: modelo treinado
- `reports/figures/xgb_poisson_beeswarm.png`: interpretação SHAP
- `reports/analytics/*.csv`: resumos analíticos da simulação

## Documentação técnica

- [01_modelo_elo](docs/01_modelo_elo.md)
- [02_schema_duckdb](docs/02_schema_duckdb.md)
- [03_poisson_dist](docs/03_poisson_dist.md)

## Qualidade de código

- Instale os hooks com `pre-commit install` e `pre-commit install --hook-type pre-push`
- Os commits passam por `ruff format` e `ruff check --fix`
- O `pytest` roda no `pre-push`



