#!/bin/bash

# OPENAI API KEY
export OPENAI_API_KEY="sk-iNPFkz9hb4QfwuqWbVBh6YoEbBrvv4yyBs5UYEFfPNbc3KeK"
export OPENAI_API_BASE="https://ai.nengyongai.cn/v1"

# ANTHROPIC API KEY
export ANTHROPIC_API_KEY=""

# HUGGING FACE API KEY
export HUGGINGFACEHUB_API_TOKEN=""

# WEATHER API KEY
export OPENWEATHERMAP_API_KEY="6211f247e50b89d06eb1a82fdede7f27"

# SMARTTHINGS API TOKEN
# Go to https://account.smartthings.com/tokens
export SMARTTHINGS_API_TOKEN=""

# Leave as empty string:
export CURL_CA_BUNDLE=""

# Determine project root based on this script's actual location.
# Using BASH_SOURCE works correctly even when this file is sourced.
BIN_FOLDER="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

export SMARTHOME_ROOT="$(dirname "$BIN_FOLDER")"
export TRIGGER_SERVER_URL="127.0.0.1:5797"
export MONGODB_SERVER_URL="127.0.0.1:27017"
