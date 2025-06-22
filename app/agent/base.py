from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Agent的抽象基类，用于管理状态和执行流程。

    提供状态转换、记忆管理、基于步骤的执行循环以及性能调优功能。
    子类必须实现 `step` 方法。
    """

    # 核心属性
    name: str = Field(..., description="Agent的唯一名称")
    description: Optional[str] = Field(None, description="Agent的可选描述")

    # 提示词
    system_prompt: Optional[str] = Field(
        None, description="系统级指令提示词"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="下一步动作提示词"
    )

    # 依赖项
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")
    memory: Memory = Field(default_factory=Memory, description="Agent的记忆存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Agent当前状态"
    )

    # 执行控制
    max_steps: int = Field(default=10, description="最大执行步数")
    current_step: int = Field(default=0, description="当前执行步数")

    duplicate_threshold: int = Field(default=2, description="检测卡壳的重复阈值")

    # 性能调优属性
    enable_result_cache: bool = Field(default=False, description="是否启用结果缓存")
    cache_ttl: int = Field(default=0, description="缓存有效期（秒）")
    preload_frequent_tools: bool = Field(default=False, description="是否预热常用工具")
    batch_size: int = Field(default=1, description="批量处理请求的数量")
    token_window_size: int = Field(default=2048, description="LLM上下文窗口大小")
    summarize_threshold: int = Field(default=0, description="历史记录压缩阈值长度")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # 允许子类定义额外字段

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """在初始化后确保LLM和Memory实例，并应用性能调优设置。"""
        # 确保 llm 和 memory 正确初始化
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()

        # 应用性能调优默认设置
        self.optimize_performance()
        return self

    def optimize_performance(self):
        """性能调优设置方法"""
        # 1. 缓存机制
        self.enable_result_cache = True
        self.cache_ttl = 3600  # 缓存有效期 3600 秒

        # 2. 工具预热
        self.preload_frequent_tools = True

        # 3. 批量处理请求
        self.batch_size = 3  # 每批处理3个请求

        # 4. 模型参数优化
        self.token_window_size = 6000  # 扩展上下文窗口到6000 tokens
        self.summarize_threshold = 7500  # 超过7500 tokens时压缩历史记录

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """状态上下文管理器，在执行期间临时切换状态。

        参数:
            new_state: 要切换到的新状态

        用法:
            async with self.state_context(AgentState.RUNNING):
                # 在 RUNNING 状态下执行代码

        异常处理:
            如果执行中抛出异常，会将状态设置为 ERROR 并重新抛出。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"无效的状态: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # 出错时切换到 ERROR 状态
            raise e
        finally:
            self.state = previous_state  # 恢复到原始状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """将消息添加到 Agent 的记忆中。"""
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"不支持的消息角色: {role}")

        # 如果是工具消息，附带额外参数
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """异步执行 Agent 的主循环。"""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"无法在 {self.state} 状态下执行 run 方法")

        # 如果有初始用户请求，先记录到记忆
        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            # 当未达到最大步数且未完成时，循环调用 step
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"执行步骤 {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # 检测卡壳并处理
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"步骤 {self.current_step}: {step_result}")

            # 超出最大步数则重置状态
            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"终止: 已达到最大步数 ({self.max_steps})")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "未执行任何步骤"

    @abstractmethod
    async def step(self) -> str:
        """执行单个步骤，需要子类实现具体逻辑。"""

    def handle_stuck_state(self):
        """处理卡壳状态，通过更新提示词改变策略。"""
        stuck_prompt = (
            "检测到重复回复。"
            "请尝试新的策略，避免重复无效路径。"
        )
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent检测到卡壳，已添加提示: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """判断Agent是否卡壳，通过检测连续重复的助手回复。"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """获取Agent记忆中的所有消息列表。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """设置Agent记忆中的消息列表。"""
        self.memory.messages = value
