# Chat with MLX

Run LLM on your Mac!

**chat-with-mlx** is based on [qnguyen3/chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx) and provides similar functionality.

A web UI is provided for using LLM chat or completion, and for using RAG search. 

In addition, plans are in place to add visual capabilities to VLM. 

## Project Status

### Functions implemented

* [x] Chat
* [x] Completion
* [ ] RAG

### To be continued

The project is currently under development.

Currently, you can only create a configuration file manually. After the creation is completed, the model file can be automatically downloaded from HuggingFace.

RAG and model management functions have not yet been implemented.

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NeptuneIsTheBest/chat-with-mlx.git
   cd chat-with-mlx
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

You can start the chat server by running the following Python script:

```bash
python app.py
```

- `--port`: The port on which the server will run (default is `7860`).
- `--share`: If specified, the server will be shared publicly.

### Model Configuration

Currently, you can create model configurations in the following ways:

1. Create a corresponding configuration file in the `./models/configs/` directory.

Here is an example configuration file for **Llama-3.2-3B-Instruct-4bit**:

**Llama-3.2-3B-Instruct-4bit.json**
```json
{
  "model_name": "Llama-3.2-3B-Instruct",
  "original_repo": "meta-llama/Llama-3.2-3B-Instruct",
  "mlx_repo": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "quantize": "4bit",
  "default_language": "multi",
  "system_prompt": "You are a useful assistant. Please reason step by step."
}
```

- `model_name`: The name of the model.
- `original_repo`: The original repository where the model can be found.
- `mlx_repo`: The repository in the MLX community.
- `quantize`: The quantization format of the model (e.g., `4bit`).
- `default_language`: Default language setting (e.g., `multi` for multilingual support).
- `system_prompt`: The initial prompt that sets the behavior of the model.

## Contributing

If you have any questions, feel free to submit an issue to discuss at any time, or if you want to contribute any code, please feel free to submit a PR.

Thanks to the maintainers of [qnguyen3/chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx), [mlx](https://github.com/ml-explore/mlx), as well as all members of the open source community, for creating such a useful library.

## License

This project is licensed under the MIT License.
