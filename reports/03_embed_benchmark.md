# Step 3a — Embedding runtime benchmark (Apple M5 Pro, MPS, fp16, max_len 512)

| model                                                      | params   |   median tokens |   p90 tokens | >512 tokens   | >1024 tokens   |   prompts/s |   est. all 30,991 (min) |
|:-----------------------------------------------------------|:---------|----------------:|-------------:|:--------------|:---------------|------------:|------------------------:|
| BAAI/bge-m3                                                | 0.57B    |              37 |          621 | 12.0%         | 5.1%           |       164.4 |                     3.1 |
| Qwen/Qwen3-Embedding-0.6B                                  | 0.60B    |              33 |          497 | 9.7%          | 2.3%           |        60.5 |                     8.5 |
| Qwen/Qwen3-Embedding-4B (extrapolated, lower bound)        | 4.0B     |             nan |          nan | nan           | nan            |       nan   |                    57   |
| nomic-ai/nomic-embed-code (7B) (extrapolated, lower bound) | 7.1B     |             nan |          nan | nan           | nan            |       nan   |                   101   |
| Qwen/Qwen3-Embedding-8B (extrapolated, lower bound)        | 7.6B     |             nan |          nan | nan           | nan            |       nan   |                   108   |
