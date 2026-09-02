import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                      
os.environ["PATH"] = "/root/miniconda3/envs/vllm/bin" + os.pathsep + os.environ["PATH"]

import time
import yaml
from typing import Literal
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


CONFIG_INFER_FILE = "configs/model_inference_schema.yaml"


class VLLMEngine:
    """
    vLLM 推理引擎，配置驱动，支持 LLM / VLM; base / SFT / LoRA / RL 等模型类型
    
    核心接口:
        engine = VLLMEngine("qwen3_4b_i")
        
        # LLM: 单轮 QA
        answer = engine.ask("What is the SMILES of water?")
        
        # VLM: 单轮 QA（纯文本）
        answer = engine.ask("Describe this image.")
        
        # VLM: 单轮图文（OCR 任务）
        answer = engine.ask_with_image("What molecules are in this image?", pil_image)
        
        # 从 dataset sample 推理
        for sample in dataset:
            answer = engine.chat_from_sample(sample, prompt_key="question")
        
        # 完整控制：自定义 messages（支持批量）
        outputs = engine.chat([
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "What is AI?"}],
        ])
    """
    
    def __init__(self, name: str, config_path: str = CONFIG_INFER_FILE):
        self.name = name
        self.config = self._load_config(name, config_path)
        self.modality: Literal["llm", "vlm"] = self.config.get("modality", "llm")
        self.llm: LLM | None = None
        self._lora_request: LoRARequest | None = None
        
    def _load_config(self, name: str, config_path: str) -> dict:
        with open(config_path, "r") as f:
            all_config = yaml.safe_load(f)
        
        vllm_config = all_config.get("vllm", {})
        model_config = vllm_config.get(name)
        if not model_config:
            available = list(vllm_config.keys())
            raise ValueError(f"Model '{name}' not found in config. Available: {available}")
        return model_config
    
    def _get_default_sampling_params(self, **overrides) -> SamplingParams:
        defaults = self.config.get("sampling_params", {})
        params = {**defaults, **overrides}
        return SamplingParams(**params)
    
    # ==================== Build ====================
    
    def build(self) -> LLM:
        model_type = self.config.get("type", "base")
        
        ### TODO: maybe only keep base/lora, sft and rl after merged can be base type
        build_methods = {
            "base": self._build_base,
            "sft": self._build_sft,
            "lora": self._build_lora,
            "rl": self._build_rl,
        }
        
        if model_type not in build_methods:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.llm = build_methods[model_type]()
        return self.llm
    
    def _build_base(self) -> LLM:
        return LLM(
            **self.config["vllm_init"],
            **self.config.get("vllm_extra", {}),
        )
    
    def _build_sft(self) -> LLM:
        return self._build_base()
    
    def _build_lora(self) -> LLM:
        lora_cfg = self.config["vllm_lora_request"]
        self._lora_request = LoRARequest(
            lora_name=lora_cfg["name"],
            lora_int_id=1,
            lora_path=lora_cfg["path"],
        )

        print(f"✅ LoRA request built: {self._lora_request=}")
        
        return LLM(
            **self.config["vllm_init"],
            **self.config.get("vllm_extra", {}),
        )
    
    def _build_rl(self) -> LLM:
        return self._build_base()
    
    # ==================== Chat ====================
    
    def chat(
        self,
        messages: list[dict] | list[list[dict]],
        sampling_params: SamplingParams | None = None,
        lora_request: LoRARequest | None = None,
        return_text: bool = True,
    ) -> list:
        """
        Args:
            messages: OpenAI 格式的消息列表
                - 单条: [{"role": "user", "content": "question"}]
                - 批量: [[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]]
                - VLM 带图: [{"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "question"}
                  ]}]
            sampling_params: 自定义采样参数，None 则使用配置默认值
            lora_request: LoRA 请求，None 则使用配置默认值或不使用 LoRA, 如果传入就override 配置中的 LoRA
            return_text: 是否只返回生成的文本，而不是完整的 RequestOutput

        Returns:
            vLLM 的 RequestOutput 列表
        """
        if self.llm is None:
            self.build()
        
        if sampling_params is None:
            sampling_params = self._get_default_sampling_params()
        
        lora_kwargs = {"lora_request": lora_request} if lora_request else {"lora_request": self._lora_request} if self._lora_request else {}
        
        outputs = self.llm.chat(
            messages,
            sampling_params=sampling_params,
            **lora_kwargs,
        )
        return [output.outputs[0].text if return_text else output for output in outputs]
    
    # ==================== 便捷方法 ====================
    
    def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        **sampling_overrides,
    ) -> str:
        """
        单轮 QA（纯文本），适用于 LLM 和 VLM。
        
        Args:
            question: 用户问题
            system_prompt: 可选的 system prompt
            **sampling_overrides: 覆盖默认 sampling params
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        sampling_params = self._get_default_sampling_params(**sampling_overrides)
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=True)
        return outputs[0]
    
    def ask_with_image(
        self,
        question: str,
        image: Image.Image,
        system_prompt: str | None = None,
        **sampling_overrides,
    ) -> str:
        """
        单轮图文 QA，仅适用于 VLM 模型。
        
        Args:
            question: 用户问题
            image: PIL Image 对象
            system_prompt: 可选的 system prompt
            **sampling_overrides: 覆盖默认 sampling params
            
        Returns:
            生成的文本
        """
        if self.modality != "vlm":
            raise ValueError(
                f"ask_with_image is only available for VLM models. "
                f"Current model '{self.name}' is '{self.modality}'."
            )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": question},
            ],
        })
        
        sampling_params = self._get_default_sampling_params(**sampling_overrides)
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=True)
        return outputs[0]
    
    def chat_from_sample(
        self,
        sample: dict,
        prompt_key: str = "prompt",
        image_key: str | None = "input_rep",
        system_prompt: str | None = None,
        **sampling_overrides,
    ) -> str:
        """
        从单个 HF dataset sample 构造 messages 并推理。
        
        Args:
            sample: HF dataset 的单条样本（dict）
            prompt_key: prompt 字段的 key
            image_key: image 字段的 key，None 表示纯文本；sample 中应为 PIL Image
            system_prompt: 可选的 system prompt
            **sampling_overrides: 覆盖默认 sampling params
            
        Returns:
            生成的文本
        """
        question = sample[prompt_key]
        image = sample.get(image_key) if image_key else None
        
        if image is not None:
            return self.ask_with_image(question, image, system_prompt, **sampling_overrides)
        else:
            return self.ask(question, system_prompt, **sampling_overrides)
    
    # ==================== Debug ====================
    
    def debug(
        self,
        question: str,
        image: Image.Image | None = None,
        system_prompt: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        调试接口，支持纯文本和图文。
        
        Args:
            question: 用户问题
            image: 可选的 PIL Image (VLM 用)
            system_prompt: 可选的 system prompt
            verbose: 是否打印调试信息
            
        Returns:
            包含 answer, messages, sampling_params, timing 等信息的字典
        """
        # 构造 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if image is not None:
            if self.modality != "vlm":
                raise ValueError(f"Image input requires VLM model, got '{self.modality}'.")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": question},
                ],
            })
        else:
            messages.append({"role": "user", "content": question})
        
        sampling_params = self._get_default_sampling_params()
        
        if verbose:
            print("=" * 60)
            print(f"[DEBUG] Model: {self.name} ({self.modality})")
            print(f"[DEBUG] Messages:")
            for msg in messages:
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image_pil":
                            print(f"  - [Image: {item['image_pil'].size}]")
                        else:
                            print(f"  - [{item['type']}]: {item['text']}")
                else:
                    print(f"  [{msg['role']}]: {content}")
            print(f"[DEBUG] Sampling params: {sampling_params}")
            print("-" * 60)
        
        # 计时推理
        start = time.time()
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=False)
        elapsed = time.time() - start
        
        result = outputs[0]
        answer = result.outputs[0].text
        prompt_tokens = len(result.prompt_token_ids) if hasattr(result, 'prompt_token_ids') else 0
        output_tokens = len(result.outputs[0].token_ids) if result.outputs[0].token_ids else 0
        
        if verbose:
            print(f"[DEBUG] Answer: {answer!r}")
            print(f"[DEBUG] Tokens: prompt={prompt_tokens}, output={output_tokens}")
            print(f"[DEBUG] Time: {elapsed:.2f}s")
            print("=" * 60)
        
        return {
            "answer": answer,
            "messages": messages,
            "sampling_params": vars(sampling_params),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "time": elapsed,
        }


if __name__ == "__main__":
    # # === LLM 用法 ===
    # print("=== LLM Example ===")
    # llm_engine = VLLMEngine("qwen3_4b_i")
    # answer = llm_engine.ask("What is the SMILES of water?")
    # print(f"Answer: {answer}")
    
    # === VLM 用法 ===
    print("\n=== VLM Example ===")
    vlm_engine = VLLMEngine("qwen3_vl_4b_i")
    
    # 纯文本 QA
    answer = vlm_engine.ask("What is a molecule?")
    print(f"Answer: {answer}")
    
    # 图文 QA（OCR 任务）
    from PIL import Image
    from urllib.request import urlopen

    img = Image.open(urlopen("https://picsum.photos/300"))
    answer = vlm_engine.ask_with_image("What is in this image?", img)
    print(f"Answer: {answer}")
    
    # 从 dataset sample 推理
    # from datasets import load_dataset
    sample = {
        "question": "Describe this image.",
        "input_rep": img,
    }
    answer = vlm_engine.chat_from_sample(sample, prompt_key="question")
    print(answer)
    
    # # 批量推理
    # all_messages = [
    #     [{"role": "user", "content": item["question"]}]
    #     for item in ds["test"].select(range(100))
    # ]
    # outputs = vlm_engine.chat(all_messages)
    
    # Debug
    result = vlm_engine.debug("Describe this image.", image=img)