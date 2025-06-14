"""
Base agent interface for Symbiote multi-agent system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SassLevel(Enum):
    """Personality/sass levels for agent responses."""
    PROFESSIONAL = 0
    POLITE = 1
    FRIENDLY = 2
    CASUAL = 3
    HUMOROUS = 4
    PLAYFUL = 5
    WITTY = 6
    SARCASTIC = 7
    SNARKY = 8
    TSUNDERE = 9
    MAXIMUM_SASS = 10


@dataclass
class AgentMessage:
    """Standard message format for agent communication."""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Configuration for agents."""
    name: str
    model_name: str
    api_key: Optional[str] = None
    sass_level: SassLevel = SassLevel.FRIENDLY
    max_tokens: int = 4096
    temperature: float = 0.7
    tools_enabled: bool = True
    memory_enabled: bool = True


class BaseAgent(ABC):
    """Base class for all Symbiote agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.sass_level = config.sass_level
        self.conversation_history: List[AgentMessage] = []
        
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process an incoming message and return a response."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent with necessary resources."""
        pass
    
    def add_to_history(self, message: AgentMessage):
        """Add a message to conversation history."""
        self.conversation_history.append(message)
    
    def get_sass_prompt(self) -> str:
        """Get sass-level appropriate prompt additions."""
        sass_prompts = {
            SassLevel.PROFESSIONAL: "Be professional and helpful.",
            SassLevel.POLITE: "Be polite and courteous.",
            SassLevel.FRIENDLY: "Be friendly and approachable.",
            SassLevel.CASUAL: "Be casual and relaxed.",
            SassLevel.HUMOROUS: "Add light humor when appropriate.",
            SassLevel.PLAYFUL: "Be playful and engaging.",
            SassLevel.WITTY: "Use wit and clever remarks.",
            SassLevel.SARCASTIC: "Be sarcastic but constructive.",
            SassLevel.SNARKY: "Be snarky and opinionated.",
            SassLevel.TSUNDERE: "Be tsundere - act tough but caring underneath.",
            SassLevel.MAXIMUM_SASS: "Full sass mode - be brutally honest and sarcastic (but still helpful)."
        }
        return sass_prompts.get(self.sass_level, sass_prompts[SassLevel.FRIENDLY])
    
    def update_sass_level(self, level: int):
        """Update the sass level (0-10)."""
        if 0 <= level <= 10:
            self.sass_level = SassLevel(level)
            return True
        return False
