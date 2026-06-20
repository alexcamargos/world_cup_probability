# World Cup Probability

Base do repositório para modelagem de probabilidades da Copa do Mundo.

## Execução do pipeline

Ordem recomendada:

1. Inicializar o warehouse e carregar dados brutos:
   - `uv run python -m src.db_init`
2. Calcular o histórico ELO:
   - `uv run python -m src.elo_engine`
3. Gerar a base de features:
   - `uv run python -m src.feature_pipeline`
4. Treinar o modelo Poisson e gerar SHAP:
   - `uv run python -m src.model`
5. Rodar a simulação Monte Carlo:
   - `uv run python -m src.simulator`
6. Gerar as análises e CSVs:
   - `uv run python -m src.analytics`
7. Orquestrar tudo em sequência:
   - `uv run python -m src.orchestrator --iterations 100000 --batch-size 2500`

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
