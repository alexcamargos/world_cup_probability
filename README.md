# World Cup Probability

Base do repositório para modelagem de probabilidades da Copa do Mundo.

## Execução do pipeline

Ordem recomendada:

1. Coletar e normalizar os dados reais usados pelo modelo:
   - `uv run python -m src.data_collection --sources matches`
   - `uv run python -m src.data_collection --sources matches --load-existing`
   - `uv run python -m src.data_collection --sources squad --ea-fc-dataset <owner/dataset>`
   - `uv run python -m src.data_collection --sources fbref --fbref-leagues <league> --fbref-seasons <season>`
   - `uv run python -m src.data_collection --sources transfermarkt --transfermarkt-manifest data/raw/transfermarkt/teams.json`
2. Inicializar o warehouse e carregar dados brutos:
   - `uv run python -m src.db_init`
3. Calcular o histórico ELO:
   - `uv run python -m src.elo_engine`
4. Gerar a base de features:
   - `uv run python -m src.feature_pipeline`
5. Treinar o modelo Poisson e gerar SHAP:
   - `uv run python -m src.model`
6. Rodar a simulação Monte Carlo:
   - `uv run python -m src.simulator`
7. Gerar as análises e CSVs:
   - `uv run python -m src.analytics`
8. Orquestrar tudo em sequência:
   - `uv run python -m src.orchestrator --iterations 100000 --batch-size 2500`

Por regra de negócio, a coleta histórica carrega apenas partidas em ou após
`2010-01-01`. O valor pode ser alterado manualmente com `--cutoff-date`.

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
