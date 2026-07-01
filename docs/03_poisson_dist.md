# Distribuição Poisson

- Objetivo: modelar contagens de gols esperados por equipe.
- Saída esperada: intensidade ofensiva e defensiva por jogo.
- Uso no pipeline: gerar probabilidades de placares e mercados derivados.

## Correção Dixon-Coles

O simulador usa as intensidades Poisson como distribuição-base de placar e
aplica a correção Dixon-Coles nos quatro placares baixos: 0 x 0, 0 x 1, 1 x 0 e
1 x 1. A ideia é relaxar a independência estrita entre gols dos dois times, que
costuma distorcer empates e resultados com poucos gols.

Para um placar `(x, y)`, lambdas `lambda_home` e `lambda_away`, e parâmetro
`rho`, o fator multiplicativo usado é:

- `0 x 0`: `1 - lambda_home * lambda_away * rho`
- `0 x 1`: `1 + lambda_home * rho`
- `1 x 0`: `1 + lambda_away * rho`
- `1 x 1`: `1 - rho`
- demais placares: `1`

O projeto calibra `rho` por validação temporal com o comando
`uv run calibrate-dixon-coles`, gravando o resultado em
`models/dixon_coles_calibration.json`. Quando esse artefato existe, `simulate`
usa o valor calibrado automaticamente. Se o artefato não existir, o fallback é
`rho = -0.10`.

Use `uv run simulate --dixon-coles-rho 0.0` para forçar Poisson independente ou
outro valor manual, ignorando o artefato calibrado.
