# Helicone Integration

## What is Helicone?

Helicone is an open-source LLM observability platform that provides access to various LLM models with:
- Request/response logging
- Cost tracking
- Performance monitoring
- Debugging tools
- Custom properties for organizing requests

Documentation: https://docs.helicone.ai/getting-started/quick-start

## Setup

### 1. Get API Key

You need **ONE API key**:

**Helicone API Key**: Get it from https://www.helicone.ai/
- Sign up for a free account
- Go to your dashboard
- Generate an API key (starts with `sk-helicone-`)

### 2. Set Environment Variable

**Windows (PowerShell)**
```powershell
$env:HELICONE_API_KEY="sk-helicone-xxxxx-xxxxx-xxxxx-xxxxx"
```

**Windows (cmd)**
```cmd
set HELICONE_API_KEY=sk-helicone-xxxxx-xxxxx-xxxxx-xxxxx
```

For permanent setup, add it to Windows Environment Variables through System Properties.

### 3. Install Dependencies

```bash
pip install -r ../requirements.txt
```

Or install individually:
```bash
pip install openai pandas
```

Required packages:
- `openai>=1.0.0` - Official OpenAI Python library (used by Helicone)
- `pandas` - For CSV handling

## How It Works

The script uses the official OpenAI Python library, but points it to Helicone's AI Gateway:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ai-gateway.helicone.ai",
    api_key=os.getenv("HELICONE_API_KEY")
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

Helicone acts as a gateway/proxy that:
1. Receives your request
2. Forwards it to the actual LLM provider
3. Logs the request/response
4. Returns the response to you

## Configuration

Edit `script_models_config.py` to configure which models to use:

```python
MODELS = [
    {"name": "gpt-4o-mini"},
    {"name": "gpt-4o"},
    {"name": "gpt-3.5-turbo"},
]
```

Available models depend on what Helicone supports. Check their documentation for the full list.

## Usage

### Input File

Place your prompts in `prompts/example.csv` with the following format:

```csv
test_item;language;prompt
B1T;English;"Your prompt here"
B1F;English;"Another prompt"
```

### Running the Script

#### Using the batch file:
```cmd
execute.bat
```

#### Or directly with Python:
```cmd
python script.py
```

### Output

- **Answers**: Saved in `answers/` directory, one CSV file per model
  - Format: `<model-name>_answers.csv`
  - Contains: test_item, language, prompt, and **answer** columns
  
- **Logs**: Saved in `logs/` directory
  - Format: `execution_YYYYMMDD_HHMMSS.log`

## Features

### Sequential Processing
- Processes prompts one by one for each model
- Saves each answer immediately after generation
- Can resume from where it stopped if interrupted

### Resume Capability
- If execution is interrupted, the script will:
  - Load existing answers
  - Skip already completed prompts
  - Continue processing pending prompts

### Error Handling
- Failed prompts are marked with "ERROR:" prefix
- Errors are logged with details
- Processing continues for other prompts

### Progress Tracking
- Real-time progress in console and logs
- Shows: completed, errors, and pending counts
- Displays test_item identifiers for easy tracking

## Helicone Dashboard

After running the script, you can:
1. Visit https://www.helicone.ai/
2. Log in to your account
3. View all requests, responses, costs, and performance metrics in the dashboard

The dashboard provides:
- Request history with full prompts and responses
- Token usage statistics
- Cost tracking
- Response time metrics
- Error monitoring

## Configuration Options

In `script.py`, you can modify:

```python
TEMPERATURE = 0.0          # Model temperature (0.0 = deterministic)
MAX_TOKENS = 200          # Maximum tokens in response
CSV_SEPARATOR = ";"        # CSV delimiter
INPUT_FILE = "prompts/example.csv"
OUTPUT_DIR = "answers"
```

## Troubleshooting

### API Key Issues
- Make sure HELICONE_API_KEY is set correctly
- The key should start with `sk-helicone-`
- No quotes or spaces in the environment variable
- Test with: `echo %HELICONE_API_KEY%` (cmd) or `echo $env:HELICONE_API_KEY` (PowerShell)

### Import Errors
If you get "No module named 'openai'":
```bash
pip install openai
```

### Connection Errors
- Check internet connection
- Verify API key is valid
- Check Helicone status: https://status.helicone.ai/

### File Encoding Issues
- Input CSV should be UTF-8 encoded
- The script handles BOM automatically

## Comparison with Other Providers

| Feature | Helicone | Ollama | HuggingFace |
|---------|----------|--------|-------------|
| **Hosting** | Cloud gateway | Local or Cloud | Cloud |
| **API Key** | Helicone key only | Ollama key | HF token |
| **Models** | Multiple providers | Ollama models | HF models |
| **Dashboard** | ✓ Full observability | ✗ No dashboard | ✓ Basic stats |
| **Cost** | Pay per use | Free (local) / Cloud pricing | Free tier + paid |
| **Setup** | Simple (1 key) | Requires installation | Account needed |

## Cost Information

Helicone itself is free for the observability features. You pay only for:
- The actual LLM API calls (charged by the model provider)
- Helicone shows you detailed cost breakdowns in the dashboard

Check Helicone's pricing page for any additional enterprise features: https://www.helicone.ai/pricing

