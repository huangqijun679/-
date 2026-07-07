# Agent Chat

一个基于 LangGraph 的智能对话助手，支持多供应商模型切换、RAG知识库、自定义技能等功能。

## 功能特性

- **多模型支持**：内置 OpenAI、DeepSeek，支持自定义 OpenAI 兼容 API 供应商
- **RAG 知识库**：上传文档或保存对话记忆，作为上下文增强回答
- **Agent 技能**：内置房贷计算、SSH 远程执行等工具
- **人物设定**：自定义系统提示词，创建不同角色
- **思考模式**：使用 DeepSeek Reasoner 进行深度推理
- **多语言支持**：中文/英文界面切换
- **暗色主题**：支持明暗主题切换

## 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/agent-chat.git
cd agent-chat

# 创建虚拟环境
python -m venv agent

# 激活虚拟环境
# Windows
agent\Scripts\activate
# Linux/macOS
source agent/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# 方式一：使用 run.py（推荐）
python run.py

# 方式二：直接启动
uvicorn main:app --host 0.0.0.0 --port 8003

# 方式三：Windows 批处理
start.bat
```

访问 http://localhost:8003 开始使用。

## 配置说明

### API 密钥配置

首次使用需要配置至少一个 API 密钥：

| 供应商 | 说明 | 密钥格式 |
|--------|------|----------|
| OpenAI | GPT-4o、GPT-3.5 等 | `sk-...` |
| DeepSeek | DeepSeek Chat、Reasoner 等 | `sk-...` |
| 自定义 | 任何 OpenAI 兼容 API | 自定义 |

### 自定义供应商

支持添加任何 OpenAI 兼容的 API 供应商：

1. 点击侧边栏「自定义模型」
2. 填写供应商信息：
   - **服务商标识**：唯一标识符（如 `my-llm`）
   - **显示名称**：界面显示名称
   - **API 接口地址**：如 `https://api.example.com/v1`
   - **模型名称**：逗号分隔的模型列表
   - **API 密钥**：对应的密钥

### RAG 知识库

两种方式增强 AI 知识：

1. **上传文档**：支持 `.txt`、`.md`、`.json`、`.csv` 格式
2. **保存对话记忆**：将历史对话保存为知识库

### 人物设定

创建自定义角色：

1. 点击侧边栏「人物」
2. 设置名称、系统提示词
3. 可选关联技能

## 内置技能

| 技能 | 说明    |
|------|-------|
| `calculate_mortgage` | 房贷计算器 |
| `bing_web_search` | 必应搜索  |
| `search_weather` | 天气搜索  |
| `get_metal_prices` | 查询今日常见贵金属 |

## 项目结构

```
agent-chat/
├── main.py              # 后端主程序（FastAPI）
├── run.py               # 启动脚本
├── start.bat            # Windows 启动脚本
├── requirements.txt     # Python 依赖
├── static/
│   ├── index.html       # 前端页面（单文件 SPA）
│   └── style.css        # 样式文件
└── data/
    └── agent.db         # SQLite 数据库
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 发送消息 |
| `/api/models` | GET | 获取所有模型 |
| `/api/active-model` | GET/POST | 获取/设置当前模型 |
| `/api/keys/{provider}` | GET/POST/DELETE | API 密钥管理 |
| `/api/keys/clear-all` | POST | 一键清除所有 API 密钥 |
| `/api/custom-providers` | GET/POST/PUT/DELETE | 自定义供应商管理 |
| `/api/personas` | GET/POST/PUT/DELETE | 人物设定管理 |
| `/api/sessions` | GET/POST/DELETE | 会话管理 |
| `/api/rag/upload` | POST | 上传 RAG 文档 |
| `/api/rag/documents` | GET/DELETE | RAG 文档管理 |
| `/api/memory/save` | POST | 保存对话记忆 |
| `/api/skills` | GET | 获取可用技能 |

## 技术栈

- **后端**：FastAPI + Uvicorn
- **前端**：原生 HTML/CSS/JavaScript（单文件）
- **AI 框架**：LangChain + LangGraph
- **数据库**：SQLite
- **嵌入模型**：text-embedding-ada-002 / deepseek-embedding

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
