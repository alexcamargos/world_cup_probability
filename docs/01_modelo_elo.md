# World Cup Probability Elo

Este projeto usa um Elo próprio para medir a força atual de cada seleção.
Ele é chamado de **World Cup Probability Elo** porque não tenta apenas contar
vitórias passadas: ele transforma o histórico recente de jogos em uma nota
dinâmica que depois entra no modelo de previsão da Copa.

Pense no Elo como uma régua simples:

- uma seleção começa com uma nota neutra;
- quando vence, tende a subir;
- quando perde, tende a cair;
- empates mexem pouco, mas também contam;
- ganhar de uma seleção forte vale mais do que ganhar de uma seleção fraca;
- perder para uma seleção fraca pesa mais do que perder para uma seleção forte.

No projeto, essa régua começa em **1500 pontos** para seleções ainda sem
histórico observado na base.

## Por Que O Projeto Usa Elo

Uma tabela comum de resultados diz apenas que um jogo terminou 2 a 1. O Elo
tenta responder uma pergunta mais útil para previsão:

> Depois desse resultado, o que aprendemos sobre a força relativa das duas
> seleções?

Essa informação é útil porque o modelo final não olha só para placares. Ele
usa várias pistas: ranking FIFA, valor de elenco, histórico de Copa, atributos
dos jogadores, estatísticas recentes e também o Elo calculado pelo projeto.

O Elo do projeto é uma dessas pistas centrais porque ele tem três vantagens:

- acompanha a evolução das seleções jogo a jogo;
- pesa a força do adversário;
- pode ser recalculado de forma reprodutível sempre que a base de jogos muda.

## De Onde Vêm Os Jogos

O cálculo usa a tabela `f_matches`, que guarda jogos internacionais históricos
carregados no banco DuckDB do projeto.

Para entrar no Elo, um jogo precisa ter:

- data;
- seleção mandante;
- seleção visitante;
- placar do mandante;
- placar do visitante;
- nome da competição, quando disponível;
- informação de campo neutro, quando disponível.

Jogos da Copa atual, quando identificados como Copa do Mundo de 2026, são
excluídos do cálculo histórico. Isso evita vazamento de informação: o modelo
não deve aprender com jogos que ele deveria prever.

## A Ordem Dos Jogos Importa

O Elo é calculado em ordem cronológica.

Isso significa que o projeto lê os jogos do mais antigo para o mais recente.
Depois de cada partida, as notas das duas seleções são atualizadas. A nota
atualizada vira o ponto de partida para o próximo jogo daquela seleção.

Esse detalhe é importante. Se Brasil joga em janeiro e Argentina joga em março,
o jogo de janeiro só pode afetar jogos que acontecem depois dele. O projeto não
usa informação do futuro para recalcular o passado.

## O Que Acontece Antes De Cada Jogo

Antes de atualizar as notas, o projeto registra a força das duas seleções
naquele momento:

- `home_rating_before`: nota do mandante antes do jogo;
- `away_rating_before`: nota do visitante antes do jogo.

Se uma seleção ainda não apareceu no histórico, ela recebe a nota inicial de
1500 pontos.

Com essas duas notas, o projeto estima qual resultado era esperado antes da
bola rolar. Uma seleção com nota maior tem expectativa maior de vitória. Se as
duas têm notas parecidas, a expectativa fica perto de meio a meio.

## Resultado Esperado E Resultado Real

O Elo compara duas coisas:

- **resultado esperado**: o que o rating dizia antes do jogo;
- **resultado real**: o que aconteceu no placar.

O resultado real é simplificado assim:

- vitória do mandante: mandante recebe `1.0`, visitante recebe `0.0`;
- empate: cada lado recebe `0.5`;
- vitória do visitante: mandante recebe `0.0`, visitante recebe `1.0`.

Exemplo simples:

- Brasil e Argentina começam com 1500 pontos;
- como as notas são iguais, o resultado esperado é equilibrado;
- se o Brasil vence, ele sobe;
- a Argentina cai na mesma medida.

Se o Brasil já tivesse rating muito maior, uma vitória normal renderia menos
pontos, porque o sistema já esperava que ele vencesse.

## Vantagem De Mando

Quando o jogo não é em campo neutro, o projeto adiciona uma vantagem temporária
ao mandante antes de calcular o resultado esperado.

Por padrão, essa vantagem de mando parte de **100 pontos de Elo**. Ela não é
somada permanentemente ao rating do time. Ela só entra na conta daquele jogo
para reconhecer que jogar em casa costuma ajudar.

Se o jogo é em campo neutro, a vantagem de mando é **0**.

Exemplo:

- mandante: 1500;
- visitante: 1500;
- jogo não neutro;
- para calcular a expectativa, o mandante é tratado como se tivesse 1600
  naquele jogo;
- depois do jogo, o rating real continua sendo atualizado normalmente.

## Peso Por Competição

Nem todo jogo ensina a mesma coisa sobre a força real de uma seleção.

Um amistoso costuma ter testes, substituições e menor pressão. Uma Copa do
Mundo costuma ter força máxima, preparação específica e maior importância. Por
isso o projeto usa pesos por tipo de competição.

Os pesos-base são:

| Tipo de competição | Peso-base |
| --- | ---: |
| Copa do Mundo | 2.5 |
| Eliminatórias de Copa | 2.0 |
| Competição continental | 1.8 |
| Nations League | 1.35 |
| Outras eliminatórias | 1.4 |
| Amistoso | 0.5 |
| Outros jogos | 1.0 |

Na prática:

- uma vitória em Copa mexe mais no Elo do que uma vitória em amistoso;
- um amistoso ainda conta, mas conta menos;
- jogos sem classificação clara usam peso neutro `1.0`.

O código também diferencia "World Cup" de "World Cup Qualifier". Isso evita que
uma eliminatória receba o mesmo peso de uma partida de Copa do Mundo.

## Margem De Gols

O Elo antigo olhava apenas para vitória, empate ou derrota. O Elo atual também
olha para a **margem de gols**.

Vencer por 4 a 0 carrega mais informação do que vencer por 1 a 0. O projeto
usa essa diferença para ajustar o tamanho da atualização.

Mas há um cuidado: goleadas de favoritos não podem inflar demais o rating.
Se uma seleção muito forte goleia uma seleção muito fraca, isso era
relativamente esperado. Se uma seleção mais fraca goleia uma mais forte, o
resultado é mais informativo.

Por isso o multiplicador de margem:

- fica em `1.0` para empate ou vitória por um gol;
- sobe quando a diferença de gols é maior;
- é limitado para evitar saltos exagerados;
- corrige o efeito quando o vencedor já era muito favorito.

## K: O Tamanho Da Atualização

No Elo, o `K` controla o quanto uma partida pode mexer no rating.

Um `K` maior significa que o sistema reage mais rápido. Um `K` menor significa
que ele é mais conservador.

No projeto, o `K` final não é fixo. Ele é montado assim:

```text
K final = K base
          x peso da competição
          x multiplicador da margem de gols
          x multiplicador de experiência
```

O `K base` parte de **20**, mas pode ser calibrado automaticamente.

## Multiplicador De Experiência

Seleções com pouco histórico observado na base têm rating mais incerto. O
projeto permite que elas se movam mais rápido no começo.

O multiplicador funciona assim:

- menos de 10 jogos observados: atualização maior;
- entre 10 e 24 jogos observados: atualização um pouco maior;
- a partir de 25 jogos: atualização normal.

Isso ajuda o rating a se ajustar mais depressa para seleções que aparecem com
poucos jogos na base, sem tornar todo o sistema instável.

## Calibração Automática

O projeto não precisa aceitar todos esses parâmetros "no chute". Ele tenta
calibrar o Elo usando o próprio histórico.

A ideia é simples:

1. O projeto separa os jogos em ordem cronológica.
2. Usa a parte mais antiga para simular a evolução dos ratings.
3. Usa a parte mais recente como validação.
4. Testa combinações de parâmetros.
5. Fica com a combinação que errou menos a expectativa dos resultados recentes.

Por padrão, a validação usa os **20% mais recentes** dos jogos históricos.

A calibração testa combinações de:

- `K base`;
- vantagem de mando;
- força da margem de gols;
- pesos por tipo de competição.

Se houver poucos jogos disponíveis, a calibração é pulada e o projeto usa os
valores-padrão. O mínimo configurado é de **80 jogos históricos**.

## O Que É Guardado No Banco

O resultado do cálculo fica na tabela `f_elo_history`.

Cada linha representa um jogo processado e guarda:

- seleção mandante e visitante;
- data e competição;
- rating das duas seleções antes do jogo;
- rating das duas seleções depois do jogo;
- resultado esperado antes do jogo;
- resultado real simplificado;
- `K` final usado na atualização;
- `K base`;
- peso da competição;
- vantagem de mando aplicada;
- margem de gols;
- multiplicador da margem de gols;
- multiplicador de experiência;
- versão da lógica de Elo;
- erro de validação da calibração, quando houver.

Essa tabela é importante porque permite auditar o caminho inteiro. Um visitante
do projeto consegue ver não apenas o rating final de uma seleção, mas também
como ela chegou até ali jogo por jogo.

## Como O Elo Entra No Modelo Final

Depois de calculado, o Elo vira uma feature do modelo de previsão.

Para cada jogo histórico, o pipeline compara o Elo do mandante com o Elo do
visitante antes da partida. Essa diferença entra no treino como:

```text
Elo do mandante antes do jogo - Elo do visitante antes do jogo
```

Se a diferença é positiva, o mandante era mais forte pelo Elo do projeto. Se é
negativa, o visitante era mais forte.

O modelo final não depende só disso. Ele combina o Elo com outras informações:

- ranking FIFA;
- World Football Elo Ratings externo;
- histórico de Copa;
- valor de elenco;
- atributos médios dos jogadores;
- estatísticas recentes, como xG e posse;
- forma recente.

O Elo do projeto é uma peça da previsão, não a previsão inteira.

## Diferença Para Outros Rankings

O projeto também usa rankings externos, como FIFA Ranking e World Football Elo
Ratings. Eles são úteis, mas têm outro papel.

O World Cup Probability Elo é calculado localmente com os dados carregados no
projeto. Isso dá controle sobre:

- quais jogos entram;
- como jogos recentes são tratados;
- como amistosos e competições oficiais são ponderados;
- como a margem de gols entra;
- como a Copa atual é excluída para evitar vazamento.

Os rankings externos entram como comparação e enriquecimento. O Elo do projeto
é a versão auditável e reproduzível dentro deste pipeline.

## Exemplo Intuitivo

Imagine três jogos:

1. Brasil vence Argentina por 1 a 0 em amistoso.
2. Brasil vence Argentina por 4 a 0 em Copa.
3. Uma seleção considerada mais fraca vence uma seleção forte por 3 a 0.

O Elo do projeto trataria esses jogos de formas diferentes:

- o amistoso conta menos;
- o jogo de Copa conta mais;
- a goleada por 4 a 0 mexe mais do que 1 a 0;
- a goleada de uma seleção mais fraca contra uma forte carrega mais surpresa e
  pode gerar ajuste maior.

Essa é a lógica central: o Elo tenta medir não só o resultado, mas o quanto o
resultado muda nossa leitura sobre a força das seleções.

## Como Rodar

O comando principal é:

```bash
uv run world-cup-probability-elo
```

Por padrão, ele tenta calibrar os parâmetros automaticamente quando há jogos
suficientes.

Para rodar sem calibração:

```bash
uv run world-cup-probability-elo --no-calibration
```

Também é possível informar parâmetros manualmente:

```bash
uv run world-cup-probability-elo \
  --base-k-factor 20 \
  --home-advantage-points 100 \
  --goal-margin-exponent 0.60
```

Depois de recalcular o Elo, os modelos devem ser retreinados para que as novas
notas entrem nas previsões.

## Resumo

O Elo do projeto é construído assim:

1. todas as seleções começam em 1500 pontos;
2. os jogos históricos são processados em ordem;
3. antes de cada jogo, o projeto estima o resultado esperado;
4. depois do placar, compara esperado contra real;
5. ajusta o rating com um `K` dinâmico;
6. esse `K` considera competição, mando, margem de gols e experiência;
7. os parâmetros podem ser calibrados automaticamente;
8. o histórico completo é salvo em `f_elo_history`;
9. o Elo vira uma feature para os modelos de previsão da Copa.

O objetivo não é dizer que uma seleção é "boa" ou "ruim" de forma absoluta. O
objetivo é manter uma medida consistente, atualizada e auditável da força
relativa das seleções ao longo do tempo.
