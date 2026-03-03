# 原生全精度大模型混合缓存加速引擎

Native full-precision LLM hybrid cache acceleration engine. **Cache is the product; model is the factory.**  
启动零资源占用，缓存命中时纯磁盘读取，未命中时按需加载 GGUF 模型推理并自动释放。

---

## 特性

- **零启动占用**：启动时不加载模型，仅校验模型并打开磁盘缓存
- **缓存即推理**：命中 L1/L2/L3/L4 时从内存或磁盘读取，不消耗 CPU/GPU
- **按需加载**：仅在缓存未命中时加载模型，推理完成后可配置空闲超时自动卸载
- **四级混合缓存**：L1(KV 内存) → L2(Logits 内存) → L3(语义磁盘) → L4(持久磁盘)，支持前缀缓存与 TTL
- **GGUF 全精度**：基于 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)，支持 Metal/CUDA/CPU，f16/bf16

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 推理后端 | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)（GGUF，Metal/CUDA） |
| 缓存存储 | 内存（L1/L2）+ [RocksDict](https://github.com/Cydemind/rocksdict)（L4）+ 本地文件（L3） |
| 语义检索 | [FAISS](https://github.com/facebookresearch/faiss)（L3 相似度缓存） |
| 哈希 | [xxhash](https://github.com/ifduyue/python-xxhash)（缓存键）、SHA-256（模型校验） |
| 配置 | PyYAML |
| CLI | prompt_toolkit、Rich |
| 安全与隐私 | cryptography（可选加密）、内容过滤接口 |
| 运行环境 | Python 3.10+，numpy |

---

## 安装

```bash
git clone https://github.com/pengjiajunwoaini-blip/ai-inference-cache-engine.git
cd ai-inference-cache-engine
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**可选**：带 Metal 的 llama-cpp-python（macOS Apple Silicon）：

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## 使用

```bash
# 指定 GGUF 模型（仅校验，不预加载）
python main.py --model /path/to/model.gguf

# 自定义上下文与 GPU 层数
python main.py --model /path/to/model.gguf --ctx-size 2048 --gpu-layers -1

# 纯 CPU
python main.py --model /path/to/model.gguf --gpu-layers 0

# 使用配置文件
python main.py --config config.yaml
```

配置文件见 `config.yaml`，可调整缓存层级、容量、驱逐策略、日志与安全等。

---

## 项目结构

```
.
├── main.py                 # 入口
├── config.yaml             # 默认配置
├── requirements.txt
├── engine/
│   ├── config.py           # 配置加载与硬件检测
│   ├── core/
│   │   ├── inference.py    # 推理引擎（按需加载/卸载）
│   │   ├── model_validator.py
│   │   └── tokenizer.py
│   ├── cache/
│   │   ├── base.py         # 缓存抽象与数据结构
│   │   ├── scheduler.py    # 多级调度
│   │   ├── l1_kv_cache.py
│   │   ├── l2_logits_cache.py
│   │   ├── l3_semantic_cache.py
│   │   ├── l4_persistent_cache.py
│   │   ├── eviction.py
│   │   └── ...
│   ├── cli/                # 交互式 CLI
│   ├── safety/             # 内容过滤、隐私
│   ├── monitoring/         # 日志、指标、健康
│   ├── verification/       # 一致性校验
│   ├── utils/              # 硬件、哈希、存储
│   └── extensions/         # 多模态、RAG、工具等
└── tests/
```

---

## 测试

```bash
pytest tests/ -v
```

---

## 许可证

MIT License. 见 [LICENSE](LICENSE)。
