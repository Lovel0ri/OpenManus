import os
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain.chat_models import ChatOpenAI
from composio_langchain import ComposioToolSet

# ——— DeepSeek / OpenAI 配置 ———
# 建议将 API Key 放到环境变量中，更安全：
# export DEEPSEEK_API_KEY="sk-xxxx"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-4753ffd4eee0442d970600e66428b3e8")
if not deepseek_api_key:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.0,
    openai_api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek OpenAI 兼容端点
)

# ——— 加载 LangChain Hub 上的函数调用 Agent 模板 ———
prompt = hub.pull("hwchase17/openai-functions-agent")

# ——— Composio 工具集配置 ———
# 建议也用环境变量保存 Composio Key：
# export COMPOSIO_API_KEY="swa9mjyu54rz2fpx8hwdh"
composio_key = os.getenv("COMPOSIO_API_KEY", "swa9mjyu54rz2fpx8hwdh")
if not composio_key:
    raise ValueError("请先设置环境变量 COMPOSIO_API_KEY")

composio_toolset = ComposioToolSet(api_key=composio_key)
tools = composio_toolset.get_tools(actions=["GMAIL_FETCH_EMAILS"])

# ——— 创建并运行 AgentExecutor ———
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ——— 执行“获取邮件”任务 ———
# 你可以指定更多过滤条件，例如“过去一周的未读邮件”等
task = "使用composio的接口，请帮我获取 Gmail 中最新的 5 封邮件的内容摘要。"
result = agent_executor.invoke({"input": task})

print(result)
