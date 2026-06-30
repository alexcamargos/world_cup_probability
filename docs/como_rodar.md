# Como rodar o projeto

Este guia descreve a ordem recomendada de execução e cada ponto de chamada
publicado em `[project.scripts]` no `pyproject.toml`.

Todos os comandos abaixo partem da raiz do repositório.

## Preparação

Requisitos:

- Python 3.12.
- `uv` instalado.
- Credenciais Kaggle para fontes que baixam datasets do Kaggle.
- Acesso de rede para fontes externas, exceto quando usar arquivos já presentes
  em `data/raw`.

Instale as dependências:

```powershell
uv sync
```

Para Kaggle, use uma das formas aceitas pela biblioteca oficial:

- `~/.kaggle/kaggle.json`
- `KAGGLE_USERNAME` e `KAGGLE_KEY`
- `KAGGLE_CONFIG_DIR` apontando para o diretório que contém `kaggle.json`

Paths padrão:

- Dados brutos: `data/raw`
- Warehouse: `data/warehouse/world_cup.duckdb`
- Modelos: `models`
- Relatórios: `reports`

## Execução recomendada

Para uma execução completa com dados atualizados:

```powershell
uv run download-data --ea-fc-dataset flynn28/eafc26-player-database
uv run load-data --ea-fc-dataset flynn28/eafc26-player-database
uv run pipeline --iterations 100000 --batch-size 2500 --seed 42
uv run dashboard
```

Para uma validação local mais rápida:

```powershell
uv run download-data --sources matches fjelstul
uv run load-data --sources matches fjelstul
uv run pipeline --iterations 1000 --batch-size 250 --seed 42
```

`pipeline` executa inicialização, Elo, ratings, ranking FIFA, features, treino,
previsões V/E/D, simulação e analytics. Ele assume que os dados brutos
necessários já existem ou que o warehouse já foi carregado; por isso,
`download-data` e `load-data` continuam sendo a preparação recomendada.

Use `--help` para ver as opções sempre que precisar:

```powershell
uv run simulate --help
```

## Fontes de dados

As fontes aceitas por `download-data`, `load-data` e `collection` são:

- `matches`: partidas internacionais via Kaggle.
- `squad`: atributos de jogadores/elencos via Kaggle.
- `fbref`: estatísticas opcionais via `soccerdata`.
- `transfermarkt`: valores de mercado por manifesto JSON.
- `fjelstul`: histórico de Copas do Mundo e disciplina.

Defaults importantes:

- `download-data` e `load-data` usam `matches`, `squad`, `transfermarkt` e
  `fjelstul` quando `--sources` não é informado.
- `collection` usa apenas `matches` por padrão.
- `fbref` é opcional e deve ser pedido explicitamente com liga e temporada, ou
  com os defaults de `config/fbref_sources.json`.
- O cutoff histórico padrão é `2010-01-01`.
- O dataset padrão para `squad` é `flynn28/eafc26-player-database`.

## Comandos de dados

### `download-data`

Baixa dados brutos para `data/raw`.

```powershell
uv run download-data
uv run download-data --sources matches
uv run download-data --sources matches squad --ea-fc-dataset flynn28/eafc26-player-database
uv run download-data --sources fbref --fbref-leagues INT-World Cup --fbref-seasons 2022
uv run download-data --sources transfermarkt --transfermarkt-manifest config/transfermarkt_teams.json
uv run download-data --sources fjelstul
```

Flags úteis:

- `--raw-dir <path>`: muda o diretório de dados brutos.
- `--force-download`: baixa de novo mesmo quando já existe arquivo local.
- `--fbref-leagues <league...>` e `--fbref-seasons <season...>`: selecionam
  fontes FBref.
- `--fbref-no-cache`: desativa cache do `soccerdata`.
- `--ea-fc-dataset <owner/dataset>`: escolhe o dataset Kaggle de jogadores.
- `--transfermarkt-manifest <path>`: arquivo JSON com seleções a coletar.

### `load-data`

Carrega dados de `data/raw` para o DuckDB.

```powershell
uv run load-data
uv run load-data --sources matches
uv run load-data --sources matches squad --ea-fc-dataset flynn28/eafc26-player-database
uv run load-data --sources fbref
uv run load-data --sources transfermarkt
uv run load-data --sources fjelstul
```

Flags úteis:

- `--db-path <path>`: muda o arquivo DuckDB.
- `--raw-dir <path>`: muda o diretório de dados brutos.
- `--cutoff-date YYYY-MM-DD`: muda a data mínima carregada.
- `--ea-fc-dataset <owner/dataset>`: escolhe qual pasta Kaggle de elenco usar.
- `--squad-season <label>`: grava o rótulo de temporada do elenco.
- `--transfermarkt-raw <path>`: escolhe um JSONL Transfermarkt específico.

### `collection`

Baixa e carrega em uma única chamada. É útil para rotinas pontuais, mas deixa
menos explícita a separação entre dados brutos e carga no warehouse.

```powershell
uv run collection --sources matches
uv run collection --sources matches --load-existing
uv run collection --sources transfermarkt --transfermarkt-manifest config/transfermarkt_teams.json
```

Flags úteis:

- `--load-existing`: não baixa da rede; carrega apenas arquivos existentes.
- `--force-download`: força novo download.
- As demais flags de fonte são equivalentes às de `download-data` e `load-data`.

### `db-init`

Cria o schema do DuckDB e carrega CSV/Parquet compatíveis encontrados em
`data/raw`.

```powershell
uv run db-init
```

Use `load-data` para as cargas específicas do projeto. Use `db-init` quando
quiser apenas garantir schema, migrações e carga genérica de arquivos locais.

## Comandos de enriquecimento

### `world-cup-probability-elo`

Calcula o histórico Elo próprio do projeto a partir de `f_matches` e grava
`f_elo_history`.

```powershell
uv run world-cup-probability-elo
```

Pré-requisito: partidas carregadas no warehouse.

### `world-football-elo-ratings`

Baixa ou recarrega um snapshot de World Football Elo Ratings e grava:

- `d_world_football_elo_ratings`
- `d_world_football_elo_team_aliases`

```powershell
uv run world-football-elo-ratings
uv run world-football-elo-ratings --load-existing --raw-path data/raw/eloratings/world_football_elo_ratings_snapshot.jsonl
uv run world-football-elo-ratings --force-download
```

Flags úteis:

- `--db-path <path>`
- `--raw-path <path>`
- `--load-existing`
- `--force-download`
- `--ratings-url <url>`
- `--team-dictionary-url <url>`

### `fifa-world-ranking`

Baixa ou recarrega um snapshot do ranking masculino da FIFA e grava:

- `d_fifa_world_ranking`
- `d_fifa_world_ranking_team_aliases`

```powershell
uv run fifa-world-ranking
uv run fifa-world-ranking --load-existing --raw-path data/raw/fifa_world_ranking/men_snapshot.jsonl
uv run fifa-world-ranking --force-download
```

Flags úteis:

- `--db-path <path>`
- `--raw-path <path>`
- `--load-existing`
- `--force-download`
- `--page-url <url>`
- `--api-url <url>`

## Comandos de modelagem

### `features`

Monta o dataframe final de features a partir do warehouse e valida se as tabelas
mínimas existem.

```powershell
uv run features
```

Este comando não grava um arquivo intermediário por padrão; ele é usado para
validar a geração de features e também é chamado pelos treinamentos.

### `train-model`

Treina o modelo XGBoost Poisson para intensidade de gols.

```powershell
uv run train-model
uv run train-model --tune --trials 50
uv run train-model --validation-fraction 0.2
uv run train-model --skip-current-world-cup-eval
```

Entradas: warehouse com partidas, Elo e features disponíveis.

Saídas principais:

- `models/xgb_poisson_model.json`
- `models/xgb_poisson_best_params.json`, quando `--tune` é usado.
- `models/world_cup_2026_holdout_metrics.json`, quando houver holdout avaliável.
- `reports/figures/xgb_poisson_beeswarm.png`

Flags úteis:

- `--db-path <path>`
- `--tune`
- `--trials <n>`
- `--timeout-seconds <n>`
- `--validation-fraction <float>`
- `--skip-current-world-cup-eval`

### `train-outcome-model`

Treina o modelo categórico de vitória, empate e derrota.

```powershell
uv run train-outcome-model
uv run train-outcome-model --tune --trials 50
uv run train-outcome-model --calibration-fraction 0.15 --validation-fraction 0.2
```

Saídas principais:

- `models/xgb_outcome_model.json`
- `models/xgb_outcome_metrics.json`
- `models/xgb_outcome_calibration.json`

Flags úteis:

- `--db-path <path>`
- `--tune`
- `--trials <n>`
- `--timeout-seconds <n>`
- `--validation-fraction <float>`
- `--calibration-fraction <float>`
- `--recency-half-life-days <float>`

### `predict-outcomes`

Gera probabilidades V/E/D para os confrontos atuais e grava a tabela
`outcome_predictions` no DuckDB, usada pelo dashboard.

```powershell
uv run predict-outcomes
uv run predict-outcomes --model-path models/xgb_outcome_model.json --calibration-path models/xgb_outcome_calibration.json
```

Flags úteis:

- `--db-path <path>`
- `--model-path <path>`
- `--calibration-path <path>`

## Comandos de simulação e análise

### `simulate`

Executa a simulação Monte Carlo da Copa do Mundo de 2026 e grava
`simulated_results`.

```powershell
uv run simulate
uv run simulate --iterations 1000 --batch-size 250 --seed 42
uv run simulate --iterations 100000 --batch-size 2500 --export-analytics
uv run simulate --disable-outcome-model
uv run simulate --dixon-coles-rho 0.0
```

Por padrão, se `models/xgb_outcome_model.json` e
`models/xgb_outcome_calibration.json` existirem, a simulação usa o modelo V/E/D
para sortear o resultado e depois amostra um placar compatível. Sem esse modelo,
usa o modo Poisson.

Flags úteis:

- `--iterations <n>`
- `--batch-size <n>`
- `--seed <n>`
- `--db-path <path>`
- `--model-path <path>`
- `--export-analytics`
- `--disable-outcome-model`
- `--dixon-coles-rho <float>`

### `analytics`

Cria views analíticas sobre `simulated_results` e exporta CSVs para
`reports/analytics`.

```powershell
uv run analytics
```

Saídas:

- `reports/analytics/semifinal_reach_probability.csv`
- `reports/analytics/title_probability.csv`
- `reports/analytics/most_probable_final_matchup.csv`

Pré-requisito: `simulate` já executado com resultados no warehouse.

### `pipeline`

Executa o fluxo de modelagem e simulação em sequência, depois que dados brutos ou
warehouse já estiverem preparados.

```powershell
uv run pipeline --iterations 100000 --batch-size 2500
uv run pipeline --iterations 1000 --batch-size 250 --seed 42
uv run pipeline --tune-model --trials 50 --timeout-seconds 1800
```

Etapas executadas:

1. Inicializa DuckDB.
2. Calcula World Cup Probability Elo.
3. Carrega World Football Elo Ratings.
4. Carrega FIFA World Ranking.
5. Monta features.
6. Treina modelo Poisson.
7. Treina modelo V/E/D.
8. Gera previsões V/E/D para dashboard.
9. Simula a Copa de 2026.
10. Exporta analytics.

Flags úteis:

- `--iterations <n>`
- `--batch-size <n>`
- `--seed <n>`
- `--db-path <path>`
- `--tune-model`
- `--trials <n>`
- `--timeout-seconds <n>`
- `--validation-fraction <float>`

## Dashboard

### `dashboard`

Abre a interface Streamlit.

```powershell
uv run dashboard
```

Alternativa equivalente:

```powershell
uv run streamlit run src/app.py
```

Pré-requisitos:

- `data/warehouse/world_cup.duckdb` existente.
- `simulated_results` gerada por `simulate` ou `pipeline`.
- `outcome_predictions` gerada por `predict-outcomes` ou `pipeline`, caso queira
  ver as probabilidades V/E/D do modelo categórico.

## Qualidade e testes

```powershell
uv run ruff format
uv run ruff check --fix
uv run sqlfluff lint src/sql
uv run sqlfluff fix src/sql
uv run pytest
```

Hooks:

```powershell
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

## Docker

O `Dockerfile` atual executa `python main.py`, que hoje é apenas um placeholder.
Ele é útil como base de empacotamento, mas o caminho documentado e testável para
rodar o pipeline neste repositório é via `uv` no ambiente local.

```powershell
docker build -t world-cup-probability .
docker run --rm world-cup-probability
```

Para usar Docker como runtime completo do pipeline, ajuste primeiro o
`Dockerfile` para instalar/expor os scripts de console do projeto e monte
`data/`, `models/`, `reports/` e credenciais Kaggle conforme necessário.
