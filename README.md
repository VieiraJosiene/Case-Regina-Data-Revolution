# Case Regina: quem recebe o e-mail?

Projeto do **Grupo 11** no **Bootcamp Data Revolution 2026 (UFJF)**. A proposta usa o histórico de uma campanha anterior para decidir quais clientes devem receber o e-mail quando o orçamento só alcança uma pequena parte da base.

## Objetivo

Uma loja virtual com cerca de 1 milhão de clientes vai lançar um produto novo por e-mail. O objetivo é **maximizar o lucro da campanha** respeitando o orçamento.

| Premissa | Valor |
| --- | --- |
| Base de clientes | 1.000.000 |
| Orçamento (cobre todos os gastos) | R$ 40.000 |
| Custo por e-mail | R$ 1 |
| Lucro por venda | R$ 100 (preço do produto: R$ 500) |
| Canal | somente e-mail |

Com R$ 1 por e-mail, o orçamento alcança no máximo 40 mil clientes, **4% da base**. A decisão é, portanto, quem recebe.

## Dados

- Arquivo: `campanha_venda_model.csv`, com 22.223 clientes da campanha anterior.
- Variável resposta: `Campanha` (1 = comprou, 0 = não comprou). Taxa geral de compra: 24,8%.
- Não há dados dos 1 milhão de clientes nem de quem não recebeu a campanha anterior.
- O arquivo **não está versionado** neste repositório (veja `.gitignore`). Coloque-o em `data/` para executar o código.

## Métodos

1. **Divisão treino/teste:** 80/20, estratificada pela variável resposta, semente 42 (4.445 clientes no teste).
2. **Modelos:** regressão logística e árvore de decisão, ambos estimando a probabilidade de compra de cada cliente.
3. **Ranking:** os clientes do teste são ordenados pela probabilidade prevista.
4. **Avaliação no topo da lista:** como o orçamento alcança 4% da base, comparamos os modelos nos 178 clientes do topo do teste (4% de 4.445), além da AUC da lista inteira.
5. **Referência:** seleção aleatória, com taxa de compra igual à da base.
6. **Cenários financeiros:** a taxa de compra do topo do teste é aplicada aos 40 mil e-mails. Resultado = compras × lucro por venda − R$ 40.000.

A árvore de decisão também foi usada para descrever perfis de compradores.

## Resultados

### Perfil de quem compra (padrões observados, não relações de causa e efeito)

- Entre os 186 clientes com engajamento 20 ou mais, 99,5% compraram.
- Clientes com menos de 40 anos compram mais (59%) do que os de 40 anos ou mais (18%).
- Mulheres compram mais (35%) do que homens (17%).
- Perfil de maior compra na árvore: até 44 anos e engajamento alto (cerca de 9 em 10 compram). Perfil de menor compra: 45 anos ou mais, pouco engajados e não mulheres (cerca de 6%).

### Comparação dos modelos (teste, 4.445 clientes)

| Métrica | Regressão logística | Árvore de decisão |
| --- | --- | --- |
| AUC (lista inteira) | 0,806 | 0,819 |
| Compradores nos 178 do topo | 164 | 154 |
| Taxa de compra no topo | 92,1% | 86,5% |
| Resultado nos 178 e-mails | R$ 16.222 | R$ 15.222 |

A árvore é um pouco melhor na lista inteira, mas a campanha só alcança o topo. Nele, **a regressão logística é a recomendada**. A diferença entre os modelos é pequena.

### Vantagem sobre a seleção aleatória (mesmos 178 e-mails)

| | Aleatória | Regressão logística |
| --- | --- | --- |
| Taxa de compra | 24,8% | 92,1% |
| Compras | ≈ 44 | 164 |
| Resultado | R$ 4.231 | R$ 16.222 (≈ 3,8×) |

### Cenários para os 40 mil e-mails

| Cenário | Taxa | Compras | Resultado (compras × R$ 100 − R$ 40.000) |
| --- | --- | --- | --- |
| Seleção aleatória | 24,8% | 9.908 | R$ 0,95 mi |
| Modelo sem engajamento (conservador) | 73,0% | 29.213 | R$ 2,88 mi |
| Modelo com engajamento | 92,1% | 36.854 | R$ 3,65 mi |

## Recomendação

1. Ordenar os clientes pela probabilidade de compra da regressão logística.
2. Enviar do topo para baixo, até o limite de e-mails do orçamento.
3. Manter um pequeno grupo sem e-mail para medir a venda realmente gerada.

## Limitações

- Os cenários supõem que os 4% do topo da base de 1 milhão se comportam como os 4% do topo do teste.
- Os resultados são históricos de teste. Não medem venda causada pelo e-mail, pois não há grupo que ficou sem receber.
- A estratégia com engajamento exige que essa variável esteja disponível antes do envio.
- Outros custos da campanha reduzem o número de e-mails e, portanto, o resultado.

## Estrutura do repositório

```
case-regina/
├── README.md
├── requirements.txt
├── vercel.json
├── public/
│   └── index.html              # página-resumo (deploy na Vercel)
├── src/
│   └── analise_campanha.py     # treino, avaliação no topo e cenários
├── notebooks/                  # notebook original do grupo (Colab)
├── data/                       # coloque aqui campanha_venda_model.csv
├── docs/                       # apresentação final e materiais
└── results/                    # saída do script (metricas.json)
```

## Como executar

Requisitos: Python 3.10 ou superior.

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

# coloque o CSV em data/campanha_venda_model.csv
python src/analise_campanha.py
```

Opções úteis:

```bash
python src/analise_campanha.py \
  --data data/campanha_venda_model.csv \
  --drop-cols ID \
  --engagement-col Engajamento \
  --tree-depth 5
```

- `--drop-cols`: colunas a ignorar, como identificadores de cliente.
- `--engagement-col`: nome da coluna de engajamento; ativa o cenário "sem engajamento".
- `--tree-depth`: profundidade máxima da árvore.

O script imprime as métricas no terminal e salva `results/metricas.json`. Os valores de premissa (orçamento, custo por e-mail e lucro por venda) ficam no início de `src/analise_campanha.py`.

> Os números das tabelas acima vêm da análise do grupo apresentada ao cliente. O script reproduz o mesmo método, mas pequenas diferenças podem aparecer se a escolha das variáveis ou os hiperparâmetros da árvore diferirem do notebook original.

## Página-resumo na Vercel

A pasta `public/` contém uma página estática com os principais resultados. Na Vercel, importe o repositório do GitHub com o preset **Other**, sem comando de build. O `vercel.json` já aponta a pasta `public` como saída.

## Equipe

Grupo 11, estudantes de Estatística da UFJF, Bootcamp Data Revolution 2026.
