# makefun-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/tiVan0/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/tiVan0/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

The best live free-model aggregator: one Node 22 server that answers OpenAI Chat Completions and Anthropic Messages from OpenRouter's fourteen free models, dialled live through make.fun's gateway. It routes each prompt to the best free model that meets its requirements (JSON, tools, streaming), fails over on refusals, and reports every upstream call.

## Run it

```sh
node --experimental-strip-types main.ts
```

Print the catalog and exit:

```sh
node --experimental-strip-types main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8xkqfxzc&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
