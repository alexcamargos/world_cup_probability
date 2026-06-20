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
5. Calcular o histórico ELO:
   - `uv run elo`
6. Gerar a base de features:
   - `uv run features`
7. Treinar o modelo Poisson e gerar SHAP:
   - `uv run train-model`
8. Rodar a simulação Monte Carlo:
   - `uv run simulate`
9. Gerar as análises e CSVs:
   - `uv run analytics`
10. Orquestrar tudo em sequência:
   - `uv run pipeline --iterations 100000 --batch-size 2500`

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
