# AIDS_PROJECT2
# Project Setup Guide

Welcome! This guide will help you set up your environment to run this project smoothly.

## 1. Create and Activate Virtual Environment

First, create a Python virtual environment to manage dependencies.

### For macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

> **Tip:** Make sure Python is installed:
> ```bash
> python --version
> ```

## 2. Install Project Dependencies

After activating your virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

## 3. Set Up Environment Variables

Create a `.env` file in the root of your project directory. Add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

Replace `your-api-key-here` with your actual OpenAI API key.

> **Important:** Never share your API keys publicly!

## 4. You're Ready!

Now you can run your Python scripts, and they will automatically use the installed dependencies and environment variables from your `.env` file.

## Optional Tips

- To update `requirements.txt` after adding new packages:
  ```bash
  pip freeze > requirements.txt
  ```

- To deactivate the virtual environment:
  ```bash
  deactivate
  ```

Happy coding! 🚀