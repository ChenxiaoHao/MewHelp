from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_base_url: str
    openai_api_key: str
    model_name: str
    history_token_budget: int = 4000
    temperature: float = 0.7

    # --- ch02: MySQL（docker compose 起本地容器，全部带默认值，不破坏 ch01 构造） ---
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""  # 2026-10-07 公开化:口令不留默认值,真值只走 gitignored .env
    mysql_db: str = "mewhelp"
    # --- ch02: 会话与工具执行 ---
    demo_user_id: str = "demo_user"
    tool_timeout_seconds: float = 10.0   # ch08 P1:executor 正名收编,基准钉 10s(原工作树未提交 30s 弃)
    tool_max_retries: int = 1

    # --- ch08: MCP 接入(P3 定死本机端口;demo 值=默认,零配置可用) ---
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersale_url: str = "http://127.0.0.1:8102/mcp"

    # --- ch05: ReAct token 预算熔断(拍板 P3;轮数上限 ch07 起并入 max_agent_steps) ---
    react_token_budget: int = 8000

    # --- ch06: 意图降级路(拍板 P2:默认空=只用大模型,配置后小判大复) ---
    intent_small_model: str = ""
    intent_confidence_threshold: float = 0.75

    # --- ch07: 三层上下文预算(拍板 P3/P4/P6;分项默认值以命中验收三数 5650/3954/1695 为校准锚) ---
    model_context_window: int = 32000
    max_output_tokens: int = 2000          # P4:软预留,不下发 max_tokens
    max_user_input_tokens: int = 2000      # P5:当前句超限 422
    max_agent_steps: int = 6               # P3:T8 接管 ReAct 轮数上限
    tool_result_max_tokens: int = 1200
    rerank_top_k: int = 5                  # P3:仅预算面证据条数,不改检索行为
    turns_to_keep: int = 20                # X:想留住的轮数
    steady_tokens_per_turn: int = 500      # Y:每轮稳态占用
    assistant_head_chars: int = 60         # 层2半压:客服答复留头字数
    summary_inject_tokens: int = 707       # 校准常数(与 S=543/证据面2500/安全1000 凑 demo 固定开销 4750)
    safety_margin_tokens: int = 1000
    history_view_messages: int = 6         # 消解/意图共用滑窗条数(=ch06 末6条现状)

    # --- ch03: RAG 知识库(全部带默认值,不破坏 ch01/ch02 构造) ---
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 80
    rag_top_k: int = 5
    rag_score_threshold: float = 0.3  # Task 12 评估集校准值
    qa_dedup_threshold: float = 0.92
    qa_mine_batch_conversations: int = 5

    # --- ch04: 混合检索+重排+评估(全部带默认值,不破坏 ch01-ch03 构造) ---
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_timeout_seconds: float = 5.0
    hybrid_recall_k: int = 50      # dense/BM25 双腿各召回 Top-50
    rrf_k: int = 60                # RRFRanker(k=60)
    rerank_top_n: int = 10         # 精排后喂给模型的证据数
    retrieval_low_conf_threshold: float = 0.161  # 闸1 终值=Task 12 回写(策略报告 D 桶校准表:误拒 4.2%/自信 3.3%)
    self_check_enabled: bool = True            # 闸2 总开关(评估对照/省调用)
    query_rewrite_enabled: bool = True         # 查询理解开关
    faith_judge_model: str = ""                # 空=model_name

    # --- ch09: Langfuse 观测(三键任一缺=观测链整体短路,行为=ch08 终态) ---
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- ch09 T4: evidence_confidence 闸定值(300 题校准择优:
    #     evals/reports/ch09_confidence_calibration.md —— sum 形,D 漏放 5.000%
    #     ≤5% · 非 D 误拦 4.167% ≤6.2% 且优于基线;改任一值须重跑校准) ---
    evidence_conf_form: str = "sum"           # rule | sum(校准脚本二选一)
    evidence_conf_floor_eff: float = 0.1      # n_eff 计数阈(标定 sum 候选所用 floor)
    evidence_conf_threshold: float = 0.168    # θ:rule=top1 下限 / sum=合分下限
    evidence_conf_n_eff_min: int = 1          # rule:有效证据条数下限(sum 形不参与)
    evidence_conf_gap_min: float = 0.0        # rule:top1-top2 间隙下限(sum 形不参与)
    evidence_conf_w_top1: float = 0.9         # sum:权重(0.9/0.05/0.05,θ=0.168)
    evidence_conf_w_n: float = 0.05
    evidence_conf_w_gap: float = 0.05
    retrieval_snapshot_text_max: int = 600    # 快照 text 截断(拍板 1A)

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """.env 文件优先于系统环境变量（spec §9: 地址/模型名/密钥全在 .env）。

        默认顺序是 init > 系统env > dotenv；此处调换成 init > dotenv > 系统env，
        避免机器上残留的 OPENAI_API_KEY 等环境变量悄悄覆盖项目配置。
        无 .env 时仍回落系统环境变量。
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
