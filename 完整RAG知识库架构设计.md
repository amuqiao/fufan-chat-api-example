# 完整RAG知识库架构设计

## 1. 系统架构概览

### 1.1 整体架构图

```mermaid
flowchart TD
    subgraph "客户端层"
        A["📱 Web客户端"]:::client
        B["💻 API客户端"]:::client
        C["🤖 第三方集成"]:::client
    end

    subgraph "API网关层"
        D["🚪 API Gateway"]:::gateway
        D -->|请求路由| E["🔐 认证授权"]:::security
        D -->|限流熔断| F["⚠️ 流量控制"]:::security
    end

    subgraph "业务逻辑层"
        G["📝 会话管理"]:::business
        G -->|历史记录| H["💬 聊天管理"]:::business
        H -->|查询处理| I["🔍 RAG核心"]:::business
        I -->|提示词构建| J["📄 提示词工程"]:::business
    end

    subgraph "RAG处理层"
        K["📚 文档加载器"]:::rag
        L["✂️ 文本分割器"]:::rag
        M["🔢 嵌入模型"]:::rag
        N["💾 向量存储"]:::rag
        O["🎯 检索器"]:::rag
        P["🔄 重排序器"]:::rag
    end

    subgraph "模型服务层"
        Q["🧠 FastChat架构"]:::model
        Q -->|模型调度| R["📊 Model Controller"]:::model
        R -->|负载均衡| S["🤖 Model Worker 1"]:::model
        R -->|负载均衡| T["🤖 Model Worker 2"]:::model
        R -->|负载均衡| U["🤖 Model Worker N"]:::model
    end

    subgraph "数据存储层"
        V["💾 向量数据库"]:::db
        W["📋 关系数据库"]:::db
        X["🗄️ 文档存储"]:::db
        Y["⏱️ 缓存服务"]:::db
    end

    subgraph "监控与评估层"
        Z["📈 性能监控"]:::monitor
        AA["📊 质量评估"]:::monitor
        AB["🔧 日志管理"]:::monitor
    end

    %% 数据流连接
    A --> D
    B --> D
    C --> D
    D --> G
    G --> H
    H --> I
    I --> J
    I --> O
    O --> P
    O --> N
    P --> I
    I --> Q
    Q --> J
    I -->|返回结果| H
    H -->|保存会话| W
    H -->|返回响应| D
    D -->|响应| A
    D -->|响应| B
    D -->|响应| C

    %% 索引构建流程
    K --> L
    L --> M
    M --> N
    N --> V
    N --> X

    %% 缓存连接
    O -->|缓存检查| Y
    Y -->|缓存命中| O
    Q -->|缓存检查| Y
    Y -->|缓存命中| Q
    H -->|缓存检查| Y
    Y -->|缓存命中| H

    %% 监控连接
    D --> Z
    G --> Z
    H --> Z
    I --> AA
    Q --> Z
    Q --> AA
    V --> Z
    W --> Z
    Y --> Z
    Z --> AB
    AA --> AB

    %% 样式定义
    classDef client fill:#FF6B6B,stroke:#2D3436,stroke-width:3px,color:white,rx:8,ry:8;
    classDef gateway fill:#4ECDC4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef security fill:#45B7D1,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8;
    classDef business fill:#96CEB4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef rag fill:#FF9FF3,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef model fill:#54A0FF,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8;
    classDef db fill:#FECA57,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef monitor fill:#E9ECEF,stroke:#2D3436,stroke-width:3px,color:#2D3436,rx:8,ry:8;
```

### 1.2 核心组件说明

| 组件 | 主要功能 | 技术选型 |
|------|---------|----------|
| **API网关** | 请求路由、认证授权、流量控制 | FastAPI + JWT + Redis |
| **会话管理** | 管理用户会话、历史记录 | Redis + MySQL |
| **聊天管理** | 处理聊天请求、构建上下文 | Python + SQLAlchemy |
| **RAG核心** | 协调检索、生成、评估流程 | 自定义Python模块 |
| **文档加载器** | 支持多种文档格式加载 | LangChain Document Loaders |
| **文本分割器** | 智能分割文本为片段 | LangChain Text Splitters |
| **嵌入模型** | 文本向量化转换 | HuggingFace Embeddings |
| **向量存储** | 向量索引构建与管理 | FAISS / Milvus |
| **检索器** | 相似文档检索 | 自定义Retriever + LangChain |
| **重排序器** | 优化检索结果排序 | BGE-Reranker |
| **FastChat架构** | 大模型管理与调度 | FastChat |
| **提示词工程** | 动态构建高质量提示词 | 自定义Prompt Templates |
| **缓存服务** | 加速检索与生成 | Redis |

## 2. 完整请求响应流程

### 2.1 用户请求处理流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Gateway as API网关
    participant Chat as 聊天管理
    participant RAG as RAG核心
    participant Cache as 缓存服务
    participant Retriever as 检索器
    participant ReRanker as 重排序器
    participant FastChat as FastChat服务
    participant DB as 数据库

    Client->>Gateway: POST /api/chat
    Gateway->>Chat: 验证会话
    Chat->>Cache: 检查查询缓存
    alt 缓存命中
        Cache-->>Chat: 返回缓存结果
        Chat->>Gateway: 返回响应
    else 缓存未命中
        Chat->>RAG: 处理查询
        RAG->>Retriever: 检索相关文档
        Retriever->>Cache: 检查检索缓存
        alt 检索缓存命中
            Cache-->>Retriever: 返回缓存的检索结果
        else 检索缓存未命中
            Retriever->>Retriever: 执行向量检索
            Retriever->>ReRanker: 重排序检索结果
            ReRanker-->>Retriever: 返回重排序结果
            Retriever->>Cache: 保存检索缓存
        end
        Retriever-->>RAG: 返回相关文档
        RAG->>RAG: 构建提示词
        RAG->>FastChat: 请求模型生成
        FastChat->>Cache: 检查生成缓存
        alt 生成缓存命中
            Cache-->>FastChat: 返回缓存的生成结果
        else 生成缓存未命中
            FastChat->>FastChat: 模型推理
            FastChat->>Cache: 保存生成缓存
        end
        FastChat-->>RAG: 返回生成结果
        RAG-->>Chat: 返回完整结果
        Chat->>DB: 保存会话历史
        Chat->>Cache: 保存查询缓存
        Chat->>Gateway: 返回响应
    end
    Gateway-->>Client: 200 OK

    Note over Client,DB: 完整请求流程: 客户端请求 → API网关 → 会话管理 → 缓存检查 → RAG处理 → 检索 → 重排序 → 模型生成 → 返回响应
```

### 2.2 文档索引构建流程

```mermaid
sequenceDiagram
    participant Admin as 管理员
    participant API as API服务
    participant Loader as 文档加载器
    participant Splitter as 文本分割器
    participant Embedder as 嵌入模型
    participant VectorDB as 向量数据库
    participant MetadataDB as 元数据数据库
    participant Cache as 缓存服务

    Admin->>API: POST /api/docs/upload
    API->>Loader: 加载文档
    Loader-->>API: 返回文档内容
    API->>Splitter: 分割文本
    Splitter-->>API: 返回文本片段
    API->>Embedder: 生成向量
    Embedder-->>API: 返回向量表示
    API->>VectorDB: 构建向量索引
    API->>MetadataDB: 存储文档元数据
    API->>Cache: 清空相关缓存
    API-->>Admin: 200 OK

    Note over Admin,Cache: 索引构建流程: 文档上传 → 加载 → 分割 → 向量化 → 向量存储 → 元数据存储 → 缓存清空
```

## 3. 缓存机制设计

### 3.1 缓存分层架构

```mermaid
flowchart TD
    subgraph "缓存分层"
        A["🔍 查询缓存层"]:::cache
        B["📚 检索结果缓存层"]:::cache
        C["🧠 生成结果缓存层"]:::cache
        D["🎯 嵌入向量缓存层"]:::cache
    end

    subgraph "缓存策略"
        E["⏱️ TTL过期策略"]:::strategy
        F["🔄 LRU淘汰策略"]:::strategy
        G["🎯 精准缓存键"]:::strategy
        H["🔊 缓存预热"]:::strategy
    end

    subgraph "缓存同步"
        I["🔄 写入时失效"]:::sync
        J["⏰ 定时刷新"]:::sync
        K["📡 事件驱动"]:::sync
    end

    A -->|缓存键: user_id:conv_id:query_hash| E
    A -->|最大容量: 10万| F
    B -->|缓存键: query_emb_hash| E
    B -->|最大容量: 100万| F
    C -->|缓存键: prompt_hash| E
    C -->|最大容量: 50万| F
    D -->|缓存键: text_hash| E
    D -->|最大容量: 500万| F

    A -->|预热: 高频查询| H
    B -->|预热: 热门文档| H
    C -->|预热: 模板提示词| H
    D -->|预热: 核心文档| H

    I -->|文档更新| B
    I -->|模型更新| C
    I -->|用户更新| A
    J -->|每天凌晨| A
    J -->|每周一| B
    K -->|事件通知| A
    K -->|事件通知| B

    classDef cache fill:#FF9FF3,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef strategy fill:#4ECDC4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef sync fill:#45B7D1,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8;
```

### 3.2 缓存键设计

| 缓存类型 | 缓存键格式 | 过期时间 | 最大容量 |
|---------|-----------|----------|----------|
| 查询缓存 | `user:{user_id}:conv:{conv_id}:query:{query_hash}` | 24小时 | 10万 |
| 检索结果缓存 | `retrieval:{query_emb_hash}:topk:{topk}` | 7天 | 100万 |
| 生成结果缓存 | `generation:{prompt_hash}:model:{model_name}` | 14天 | 50万 |
| 嵌入向量缓存 | `embedding:{text_hash}:model:{model_name}` | 30天 | 500万 |
| 会话缓存 | `session:{user_id}:conv:{conv_id}` | 1小时 | 100万 |

### 3.3 缓存实现示例

```python
# 缓存服务实现
import redis
import hashlib
import json
from typing import Any, Optional

class CacheService:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def _generate_hash(self, data: Any) -> str:
        """生成数据的哈希值作为缓存键"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """获取缓存数据"""
        try:
            value = self.redis_client.get(cache_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, cache_key: str, value: Any, expire_seconds: int) -> bool:
        """设置缓存数据"""
        try:
            self.redis_client.setex(cache_key, expire_seconds, json.dumps(value, ensure_ascii=False))
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, cache_key: str) -> bool:
        """删除缓存数据"""
        try:
            self.redis_client.delete(cache_key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> bool:
        """删除匹配模式的缓存数据"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            print(f"Cache delete pattern error: {e}")
            return False
```

## 3. FastChat模型管理设计

### 3.1 FastChat架构集成

```mermaid
flowchart TD
    subgraph "FastChat服务层"
        A["🔄 API适配器"]:::fastchat
        A -->|请求转换| B["📊 FastChat Controller"]:::fastchat
        B -->|注册管理| C["🤖 Model Worker 1"]:::fastchat
        B -->|注册管理| D["🤖 Model Worker 2"]:::fastchat
        B -->|注册管理| E["🤖 Model Worker N"]:::fastchat
        B -->|负载均衡| F["⚖️ 请求调度"]:::fastchat
    end

    subgraph "模型部署层"
        C -->|加载模型| G["📦 ChatGLM3-6B"]:::model
        D -->|加载模型| H["📦 GLM4-9B-Chat"]:::model
        E -->|加载模型| I["📦 LLaMA3-70B"]:::model
    end

    subgraph "RAG集成层"
        J["🔗 模型调用接口"]:::integration
        J -->|生成请求| A
        A -->|生成结果| J
        J -->|流式输出| K["📡 SSE处理"]:::integration
    end

    classDef fastchat fill:#54A0FF,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8;
    classDef model fill:#FF6B6B,stroke:#2D3436,stroke-width:3px,color:white,rx:8,ry:8;
    classDef integration fill:#96CEB4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
```

### 3.2 FastChat服务启动配置

```bash
# 1. 启动Controller
gunicorn -w 1 -b 0.0.0.0:20001 fastchat.serve.controller:app

# 2. 启动Model Worker (ChatGLM3-6B)
python -m fastchat.serve.model_worker \
    --model-path /path/to/chatglm3-6b \
    --model-names chatglm3-6b \
    --controller http://localhost:20001 \
    --port 20002 \
    --device cuda

# 3. 启动Model Worker (GLM4-9B-Chat)
python -m fastchat.serve.model_worker \
    --model-path /path/to/glm4-9b-chat \
    --model-names glm4-9b-chat \
    --controller http://localhost:20001 \
    --port 20003 \
    --device cuda

# 4. 启动OpenAI API服务
python -m fastchat.serve.openai_api_server \
    --controller http://localhost:20001 \
    --host 0.0.0.0 \
    --port 8000
```

### 3.3 FastChat模型调用实现

```python
# FastChat模型服务调用实现
import requests
import json
from typing import List, Dict, Optional

class FastChatService:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def generate(self, messages: List[Dict], model: str = "chatglm3-6b", 
                temperature: float = 0.7, max_tokens: int = 1024, 
                stream: bool = False) -> Dict:
        """调用FastChat生成文本"""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"FastChat generate error: {e}")
            return {"error": str(e)}
    
    def get_models(self) -> List[str]:
        """获取可用模型列表"""
        url = f"{self.base_url}/v1/models"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except Exception as e:
            print(f"FastChat get models error: {e}")
            return []

# 示例使用
fastchat_service = FastChatService()
messages = [
    {"role": "system", "content": "你是一个帮助用户的助手。"},
    {"role": "user", "content": "什么是RAG？"}
]
response = fastchat_service.generate(messages, model="chatglm3-6b")
print(response["choices"][0]["message"]["content"])
```

## 4. 关键组件实现示例

### 4.1 RAG核心处理实现

```python
# RAG核心处理实现
class RAGCore:
    def __init__(self, retriever, reranker, fastchat_service, cache_service, prompt_template):
        self.retriever = retriever
        self.reranker = reranker
        self.fastchat_service = fastchat_service
        self.cache_service = cache_service
        self.prompt_template = prompt_template
    
    def process_query(self, query: str, user_id: str, conv_id: str, model_name: str = "chatglm3-6b") -> Dict:
        """处理用户查询，执行完整RAG流程"""
        # 1. 检查查询缓存
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"user:{user_id}:conv:{conv_id}:query:{query_hash}"
        cached_result = self.cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        # 2. 执行检索
        retrieval_cache_key = f"retrieval:{query_hash}:topk:5"
        retrieved_docs = self.cache_service.get(retrieval_cache_key)
        
        if not retrieved_docs:
            # 2.1 向量检索
            retrieved_docs = self.retriever.retrieve(query, topk=10)
            # 2.2 重排序
            retrieved_docs = self.reranker.rerank(query, retrieved_docs, topk=5)
            # 2.3 保存检索缓存
            self.cache_service.set(retrieval_cache_key, retrieved_docs, expire_seconds=7*24*3600)
        
        # 3. 构建提示词
        prompt = self.prompt_template.build_prompt(query, retrieved_docs)
        
        # 4. 调用模型生成
        messages = [
            {"role": "system", "content": "你是一个基于给定文档回答问题的助手。"},
            {"role": "user", "content": prompt}
        ]
        
        # 4.1 检查生成缓存
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        generation_cache_key = f"generation:{prompt_hash}:model:{model_name}"
        generated_result = self.cache_service.get(generation_cache_key)
        
        if not generated_result:
            # 4.2 调用FastChat生成
            generated_result = self.fastchat_service.generate(messages, model=model_name)
            # 4.3 保存生成缓存
            self.cache_service.set(generation_cache_key, generated_result, expire_seconds=14*24*3600)
        
        # 5. 构建最终结果
        result = {
            "query": query,
            "answer": generated_result["choices"][0]["message"]["content"],
            "retrieved_docs": retrieved_docs,
            "model": model_name,
            "timestamp": int(time.time())
        }
        
        # 6. 保存查询缓存
        self.cache_service.set(cache_key, result, expire_seconds=24*3600)
        
        return result
```

### 4.2 提示词模板实现

```python
# 提示词模板实现
class PromptTemplate:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """根据查询和检索结果构建提示词"""
        # 构建参考文档部分
        references = ""
        for i, doc in enumerate(retrieved_docs):
            references += f"[{i+1}] {doc['content']}\n\n"
        
        # 构建完整提示词
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"参考文档：\n{references}\n"
        prompt += f"问题：{query}\n"
        prompt += "请基于参考文档回答问题，保持答案简洁准确。如果参考文档中没有相关信息，请说'根据提供的参考文档，无法回答该问题。'\n"
        prompt += "回答："
        
        return prompt
```

## 5. 性能优化策略

### 5.1 检索优化
- **向量索引优化**：使用FAISS的IVF索引或HNSW索引加速检索
- **批次检索**：支持批量查询处理，提高吞吐量
- **多级检索**：先使用粗检索过滤大量文档，再使用精检索优化结果
- **检索缓存**：缓存高频查询的检索结果

### 5.2 生成优化
- **模型量化**：使用8-bit或4-bit量化减少模型内存占用
- **模型并行**：对于大模型，使用张量并行或流水线并行加速推理
- **生成缓存**：缓存高频提示词的生成结果
- **流式输出**：支持SSE流式输出，提高用户体验

### 5.3 系统优化
- **异步处理**：使用异步框架处理并发请求
- **连接池**：数据库和Redis连接池管理
- **负载均衡**：FastChat Controller的请求调度优化
- **监控告警**：实时监控系统性能，及时发现并处理问题

## 6. 监控与评估

### 6.1 监控指标

| 指标类型 | 具体指标 | 监控频率 |
|---------|---------|----------|
| **系统性能** | API响应时间、并发请求数、错误率 | 每秒 |
| **检索性能** | 检索延迟、检索准确率、召回率 | 每分钟 |
| **生成性能** | 生成延迟、生成质量、token生成速度 | 每分钟 |
| **缓存性能** | 缓存命中率、缓存大小、缓存更新频率 | 每分钟 |
| **资源利用率** | CPU使用率、内存使用率、GPU使用率 | 每5秒 |

### 6.2 质量评估

```mermaid
flowchart TD
    A["📝 生成结果"]:::eval
    B["🔍 相关性评估"]:::eval
    C["📊 准确性评估"]:::eval
    D["💬 流畅性评估"]:::eval
    E["⚖️ 公平性评估"]:::eval
    F["📋 评估报告"]:::eval
    
    A --> B
    A --> C
    A --> D
    A --> E
    B --> F
    C --> F
    D --> F
    E --> F
    F -->|优化建议| G["🔧 系统优化"]:::eval
    
    classDef eval fill:#E9ECEF,stroke:#2D3436,stroke-width:3px,color:#2D3436,rx:8,ry:8;
```

### 6.3 评估实现示例

```python
# 简单的生成质量评估实现
class Evaluator:
    def __init__(self, model_name: str = "chatglm3-6b"):
        self.model_name = model_name
        self.fastchat_service = FastChatService()
    
    def evaluate_relevance(self, query: str, answer: str, retrieved_docs: List[Dict]) -> float:
        """评估回答与查询的相关性"""
        # 构建评估提示词
        prompt = f"评估以下回答与问题的相关性，使用0-10分评分，0分完全不相关，10分完全相关。\n"
        prompt += f"问题：{query}\n"
        prompt += f"回答：{answer}\n"
        prompt += "请只返回数字评分，不要添加任何其他内容。"
        
        messages = [{"role": "user", "content": prompt}]
        response = self.fastchat_service.generate(messages, model=self.model_name)
        
        try:
            score = float(response["choices"][0]["message"]["content"].strip())
            return score / 10.0  # 转换为0-1范围
        except:
            return 0.0
    
    def evaluate_accuracy(self, answer: str, ground_truth: str) -> float:
        """评估回答的准确性"""
        # 构建评估提示词
        prompt = f"评估以下回答与标准答案的准确性，使用0-10分评分，0分完全错误，10分完全正确。\n"
        prompt += f"标准答案：{ground_truth}\n"
        prompt += f"回答：{answer}\n"
        prompt += "请只返回数字评分，不要添加任何其他内容。"
        
        messages = [{"role": "user", "content": prompt}]
        response = self.fastchat_service.generate(messages, model=self.model_name)
        
        try:
            score = float(response["choices"][0]["message"]["content"].strip())
            return score / 10.0  # 转换为0-1范围
        except:
            return 0.0
```

## 7. 部署架构

### 7.1 单机部署架构

适合开发和测试环境，所有组件部署在同一台机器上：

```mermaid
flowchart TD
    subgraph "主机服务器"
        A["🔌 API网关"]:::gateway
        B["🔍 RAG核心"]:::business
        C["🧠 FastChat"]:::model
        D["💾 向量存储"]:::db
        E["⏱️ 缓存服务"]:::db
        F["📋 数据库"]:::db
        G["🗄️ 文档存储"]:::db
        H["📊 监控系统"]:::monitor
    end

    A -->|请求处理| B
    B -->|模型调用| C
    B -->|向量检索| D
    B -->|缓存查询| E
    B -->|数据持久化| F
    B -->|文档访问| G
    H -->|监控| A & B & C & D & E & F & G

    classDef gateway fill:#4ECDC4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef business fill:#96CEB4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef model fill:#54A0FF,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8;
    classDef db fill:#FECA57,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef monitor fill:#E9ECEF,stroke:#2D3436,stroke-width:3px,color:#2D3436,rx:8,ry:8;
```

### 7.2 分布式部署架构

适合生产环境，组件分布式部署，提高系统可靠性和扩展性：

```mermaid
flowchart TD
    subgraph "负载均衡层"
        A["🚪 Nginx 1"]:::gateway
        B["🚪 Nginx 2"]:::gateway
        C["🚪 Nginx 3"]:::gateway
        D["📊 流量监控"]:::monitor
    end

    subgraph "业务集群 1"
        E["🔌 API网关集群 1"]:::gateway
        F["🔍 RAG核心集群 1"]:::business
    end

    subgraph "业务集群 2"
        G["🔌 API网关集群 2"]:::gateway
        H["🔍 RAG核心集群 2"]:::business
    end

    subgraph "模型服务层"
        I["📊 FastChat Controller"]:::model
        J["🤖 Model Worker 1"]:::model
        K["🤖 Model Worker 2"]:::model
        L["🤖 Model Worker N"]:::model
    end

    subgraph "数据存储层"
        M["💾 向量数据库集群"]:::db
        N["📋 关系数据库集群"]:::db
    end

    A & B & C -->|负载均衡| E & G
    D -->|监控| A & B & C
    E -->|请求处理| F
    G -->|请求处理| H
    F & H -->|模型调用| I
    I -->|调度请求| J & K & L
    J & K & L -->|向量查询| M
    F & H -->|数据存储| N
    I -->|元数据存储| N

    classDef gateway fill:#4ECDC4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef business fill:#96CEB4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef model fill:#54A0FF,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8;
    classDef db fill:#FECA57,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8;
    classDef monitor fill:#E9ECEF,stroke:#2D3436,stroke-width:3px,color:#2D3436,rx:8,ry:8;
```

## 8. 总结

本设计提供了一个完整的RAG知识库架构，具有以下特点：

1. **模块化设计**：清晰的组件划分，便于维护和扩展
2. **完整的请求响应流程**：从客户端请求到模型生成的全流程设计
3. **多层缓存机制**：检索缓存、生成缓存、嵌入缓存等，加速系统响应
4. **FastChat集成**：成熟的大模型管理和调度架构
5. **性能优化策略**：检索优化、生成优化、系统优化多管齐下
6. **监控与评估**：全面的监控指标和质量评估机制
7. **灵活的部署架构**：支持单机和分布式部署

这个架构设计可以作为构建生产级RAG知识库系统的参考，根据具体业务需求和资源情况进行调整和优化。