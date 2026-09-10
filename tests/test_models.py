import unittest
from unittest.mock import Mock, patch

import torch

from src.config import LLM_MODEL
from src.models import _create_pipeline, _load_model


class ModelSetupTest(unittest.TestCase):
    @patch("src.models.find_spec", return_value=None)
    @patch("src.models.torch.cuda.is_available", return_value=True)
    def test_gpu_requires_bitsandbytes(self, _cuda_available, _find_spec):
        with self.assertRaisesRegex(RuntimeError, "requirements-gpu.txt"):
            _load_model()

    @patch("src.models.AutoModelForCausalLM.from_pretrained")
    @patch("src.models.BitsAndBytesConfig")
    @patch("src.models.find_spec", return_value=object())
    @patch("src.models.torch.cuda.is_bf16_supported", return_value=True)
    @patch("src.models.torch.cuda.is_available", return_value=True)
    def test_gpu_uses_quantization_and_automatic_device_map(
        self,
        _cuda_available,
        _bf16_supported,
        _find_spec,
        config_factory,
        model_factory,
    ):
        config = Mock()
        config_factory.return_value = config

        _load_model()

        config_factory.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_factory.assert_called_once_with(
            LLM_MODEL,
            quantization_config=config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    @patch("src.models.AutoModelForCausalLM.from_pretrained")
    @patch("src.models.BitsAndBytesConfig")
    @patch("src.models.find_spec", return_value=object())
    @patch("src.models.torch.cuda.is_bf16_supported", return_value=False)
    @patch("src.models.torch.cuda.is_available", return_value=True)
    def test_gpu_falls_back_to_float16(
        self,
        _cuda_available,
        _bf16_supported,
        _find_spec,
        config_factory,
        _model_factory,
    ):
        _load_model()

        self.assertEqual(
            config_factory.call_args.kwargs["bnb_4bit_compute_dtype"], torch.float16
        )

    @patch("src.models.AutoModelForCausalLM.from_pretrained")
    @patch("src.models.torch.cuda.is_available", return_value=False)
    def test_cpu_uses_bfloat16(self, _cuda_available, model_factory):
        _load_model()

        model_factory.assert_called_once_with(
            LLM_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    @patch("src.models.HuggingFacePipeline")
    @patch("src.models.pipeline")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_pipeline_uses_sentencepiece_tokenizer(
        self,
        tokenizer_factory,
        pipeline_factory,
        wrapper_factory,
    ):
        model = Mock()
        tokenizer = Mock(eos_token_id=2)
        tokenizer_factory.return_value = tokenizer
        generated_pipeline = Mock()
        pipeline_factory.return_value = generated_pipeline

        _create_pipeline(model)

        tokenizer_factory.assert_called_once_with(LLM_MODEL, use_fast=False)
        pipeline_factory.assert_called_once_with(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            pad_token_id=2,
            return_full_text=False,
        )
        wrapper_factory.assert_called_once_with(pipeline=generated_pipeline)


if __name__ == "__main__":
    unittest.main()
