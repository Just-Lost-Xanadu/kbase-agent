把你要喂给 Agent 的业务文档放进本目录（`.md` / `.txt` 均可，`app/retrieval/loader.py` 负责解析；`.pdf` 需要额外解析库，列为后续扩展）。

仓库已提供两份示例文档：`员工手册_示例.md` 与 `产品FAQ_示例.md`，评测集 `eval/questions.jsonl` 的 10 条问题就基于它们编写——想换成你自己的领域时，文档与评测要一起换。
