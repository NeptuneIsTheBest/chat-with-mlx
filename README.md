# Chat with MLX

> [!Warning]
> The project is currently under active development, please update frequently using the following commands: `pip install -U git+https://github.com/NeptuneIsTheBest/chat-with-mlx.git`

**Run LLM on your Mac!** An all-in-one chat Web UI based on the MLX framework, designed for Apple Silicon.

The idea of chat-with-mlx comes from [qnguyen3/chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx).

chat-with-mlx provides a similar and more modern experience, and offers more features.

You can **upload files**, or even **images** to chat when using the vision model.

If this helps you, I'd be happy if you could give me a star, thank you. ✨

## TL;DR

Use the following commands to install and run:

```bash
python -m venv chat-with-mlx
cd chat-with-mlx
. ./bin/activate
pip install git+https://github.com/NeptuneIsTheBest/chat-with-mlx.git
chat-with-mlx
```

## Roadmap

### Key Features

* [x] Chat
* [x] Completion
* [x] Model Management
* [ ] RAG

### Others

* [x] Upload file to chat（support `PDF, Word, Excel, PPT` and some plain text file like `.txt, .csv, .md`.）
* [x] Upload picture to chat (Currently tested on the `Phi-3.5-vision-instruct` model)
* [ ] and so on...

## How to use

### Installation

1. Install using pip:
   ```bash
   python -m venv chat-with-mlx
   cd chat-with-mlx
   . ./bin/activate
   pip install git+https://github.com/NeptuneIsTheBest/chat-with-mlx.git
   ```

### Run

2. Start the server:
   ```bash
   chat-with-mlx
   ```

- `--port`: The port on which the server will run (default is `7860`).
- `--share`: If specified, the server will be shared publicly.

### Use

3. Use in browser: By default, a page will open, http://127.0.0.1:7860, where you can chat.

### Model Configuration

You no longer need to add models by manually editing configuration files, you only need to use the "Model Management" page to add your models.

You can add various models from [mlx-community](https://huggingface.co/mlx-community). Models will be automatically downloaded from HuggingFace.

For the following configuration files, the model files will be stored in `models/models/Ministral-8B-Instruct-2410-4bit`.

**Ministral-8B-Instruct-2410-4bit.json**

```json
{
  "original_repo": "mistralai/Ministral-8B-Instruct-2410",
  "mlx_repo": "mlx-community/Ministral-8B-Instruct-2410-4bit",
  "model_name": "Ministral-8B-Instruct-2410-4bit",
  "quantize": "4bit",
  "default_language": "multi",
  "system_prompt": "",
  "multimodal_ability": []
}
```

- `original_repo`: The original repository where the model can be found.
- `mlx_repo`: The repository in the MLX community.
- `model_name`: The name of the model.
- `quantize`: The quantization format of the model (e.g., `4bit`).
- `default_language`: Default language setting (e.g., `multi` for multilingual support).
- `system_prompt`: The system prompt of the model.
- `multimodal_ability`: The multimodal capabilities of the model.

## Contributing

If you have any questions, feel free to submit an issue to discuss at any time, or if you want to contribute any code, please feel free to submit a PR.

Thanks to the maintainers of [qnguyen3/chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx), [mlx](https://github.com/ml-explore/mlx), as well as all members of the open source community, for
creating such a useful library.

## License

This project is licensed under the MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NeptuneIsTheBest/chat-with-mlx&type=Date)](https://star-history.com/#NeptuneIsTheBest/chat-with-mlx&Date)