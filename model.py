import enum
import gc
from dataclasses import dataclass
from typing import Dict, Union, List

from mlx_lm import load, generate, stream_generate


class MessageRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> Dict:
        return {"role": self.role.value, "content": self.content}


class Model:
    def __init__(self, model_path: str, max_tokens: int = 4096, ):
        self.model_path = model_path
        self.max_tokens = max_tokens

        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        try:
            # Replace with actual model loading code
            model, tokenizer = load(self.model_path)
            return model, tokenizer
        except Exception as e:
            raise e

    def generate_response(
            self,
            message: Dict,
            history: Union[str, List[Dict]],
            stream: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 512,
            repetition_penalty: float = 1.0
    ):
        message = [Message(MessageRole.USER, message["text"]).to_dict()]

        conversation = history + message

        formatted_prompt = self.tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=True)

        try:
            if stream:
                return self._stream_generate(
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
            else:
                return self._generate(
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
        except Exception as e:
            raise e

    def _generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            temp=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

    def _stream_generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        return stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            temp=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
