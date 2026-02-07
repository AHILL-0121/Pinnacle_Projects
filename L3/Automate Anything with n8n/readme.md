<h1 class="title">AI-Powered Content Creator Agent</h1>

<style>
.title {
  text-align: center;
  animation: fadeIn 1.5s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>


![n8n](https://img.shields.io/badge/n8n-Workflow-EA4B71?style=for-the-badge&logo=n8n)
![Ollama](https://img.shields.io/badge/Ollama-Local_AI-000000?style=for-the-badge)
![Tavily](https://img.shields.io/badge/Tavily-Search_API-4285F4?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

**An intelligent, no-code automation system that researches trending topics and generates platform-specific content for LinkedIn, X (Twitter), and blogs.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Workflow Breakdown](#-workflow-breakdown)
- [Google Sheets Setup](#-google-sheets-setup)
- [API Keys & Credentials](#-api-keys--credentials)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Security Best Practices](#-security-best-practices)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

The **AI-Powered Content Creator Agent** is a fully automated n8n workflow that eliminates manual content creation by:

1. **Reading topics** from a Google Sheet
2. **Researching** current information via Tavily Search API
3. **Generating** platform-optimized content using local Llama 3.1 AI
4. **Publishing** results back to Google Sheets
5. **Scheduling** automatic runs every 6 hours

This system is designed for content marketers, social media managers, and businesses seeking scalable, high-quality content generation without manual intervention.

---

## ✨ Features

### 🚀 Core Capabilities

- ✅ **Automated Research**: Real-time web scraping via Tavily API
- ✅ **Multi-Platform Content**: LinkedIn posts, X tweets, and blog summaries
- ✅ **Local AI Processing**: Privacy-focused using Ollama (Llama 3.1)
- ✅ **Smart Scheduling**: Runs every 6 hours automatically
- ✅ **Status Tracking**: Automatic "Pending" → "Completed" workflow
- ✅ **Error Handling**: Graceful failures with retry logic
- ✅ **Scalable**: Processes multiple topics sequentially

### 🎨 Content Specifications

| Platform | Length | Tone | Hashtags |
|----------|--------|------|----------|
| **LinkedIn** | 120-200 words | Professional, insightful | None |
| **X (Twitter)** | Max 280 characters | Concise, engaging | None |
| **Blog** | 150-200 words | Informative, neutral | N/A |

---

## 🏗️ Architecture

```
┌─────────────────┐
│ Schedule Trigger│ (Every 6 hours)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Google Sheets   │ ← Read all rows
│ (Read Topics)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Filter Pending  │ ← Status = "Pending"
│ Topics          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Loop Over       │ ← Process 1 topic/iteration
│ Topics          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tavily Research │ ← Web search (5 sources)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Split & Aggregate│ ← Combine research data
│ Results         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prepare Data    │ ← Format for AI models
└────────┬────────┘
         │
         ├────────────────┬────────────────┐
         ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LinkedIn Gen │  │   X Gen      │  │  Blog Gen    │
│ (Ollama)     │  │  (Ollama)    │  │  (Ollama)    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                ▼                 
         ┌─────────────┐
         │    Merge    │ ← Combine outputs
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │ Format Data │ ← Add timestamp, status
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │ Update Sheet│ ← Write to Google Sheets
         └──────┬──────┘
                │
                ▼
         (Loop back to next topic)
```
## 🔄WORKFLOW
![Architecture Diagram](https://i.postimg.cc/FHzRYjRc/AI-Content-Creator-Agent-Workflow.png "System Architecture")
---

## 📦 Prerequisites

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| **n8n** | v1.0+ | Workflow automation platform |
| **Ollama** | Latest | Local AI model runtime |
| **Llama 3.1** | 8B+ | Language model for content generation |
| **Node.js** | 18+ | n8n runtime environment |

### Required API Keys

- **Tavily Search API**: [Get API Key](https://tavily.com)
- **Google Sheets API**: OAuth2 credentials via Google Cloud Console

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for Llama 3.1)
- **Storage**: 10GB free space for Ollama models
- **Network**: Stable internet for API calls

---

## 🚀 Installation

### Step 1: Install n8n

```bash
# Using npm (recommended)
npm install -g n8n

# Or using Docker
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

### Step 2: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### Step 3: Pull Llama 3.1 Model

```bash
ollama pull llama3.1
```

### Step 4: Verify Ollama is Running

```bash
# Start Ollama server
ollama serve

# Test in another terminal
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Hello world",
  "stream": false
}'
```

### Step 5: Import Workflow

1. Start n8n:
   ```bash
   n8n
   ```
2. Open browser: `http://localhost:5678`
3. Navigate to **Workflows** → **Import from File**
4. Upload the `workflow.json` file from this repository
5. Activate the workflow

---

## ⚙️ Configuration

### Google Sheets Setup

#### 1. Create Spreadsheet

Create a Google Sheet with the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Topic` | Text | ✅ | Content topic/keyword |
| `Status` | Text | ✅ | "Pending" or "Completed" |
| `LinkedIn_Post` | Text | ❌ | Generated LinkedIn content |
| `X_Post` | Text | ❌ | Generated X/Twitter content |
| `Blog_Summary` | Text | ❌ | Generated blog summary |
| `Published_Date` | DateTime | ❌ | Auto-generated timestamp |

#### 2. Sample Data

```
Topic                              | Status
-----------------------------------|----------
AI Trends in Healthcare (2025)     | Pending
Future of Remote Work              | Pending
Sustainable Energy Solutions       | Pending
```

#### 3. Share Sheet

- Get the **Spreadsheet ID** from URL:
  ```
  https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
  ```
- Share with your Google Service Account email

### n8n Node Configuration

#### Read Pending Topics Node

```javascript
Document ID: YOUR_SPREADSHEET_ID
Sheet Name: Sheet1
```

#### Tavily Research Node

```javascript
API Key: YOUR_TAVILY_API_KEY
Max Results: 5
Search Depth: advanced
```

#### Generate Content Nodes (LinkedIn/X/Blog)

```javascript
URL: http://127.0.0.1:11434/api/generate
Model: llama3.1
Stream: false
```

---

## 📖 Usage

### Manual Execution

1. Add topics to Google Sheet with `Status = "Pending"`
2. Open n8n workflow
3. Click **Execute Workflow** button
4. Monitor execution in real-time
5. Check Google Sheet for generated content

### Automatic Scheduling

The workflow runs automatically every 6 hours:

- **00:00** (Midnight)
- **06:00** (Morning)
- **12:00** (Noon)
- **18:00** (Evening)

### Modifying Schedule

Edit the **Schedule Trigger** node:

```javascript
// Every 3 hours
Interval: 3 hours

// Daily at 9 AM
Cron Expression: 0 9 * * *

// Weekdays only at 10 AM
Cron Expression: 0 10 * * 1-5
```

---

## 🔍 Workflow Breakdown

### Node-by-Node Explanation

#### 1️⃣ Schedule Trigger
- **Purpose**: Initiates workflow execution
- **Frequency**: Every 6 hours
- **Output**: Timestamp

#### 2️⃣ Read Pending Topics
- **Purpose**: Fetches all rows from Google Sheets
- **API**: Google Sheets API v4
- **Output**: Array of topic objects

#### 3️⃣ Filter Pending Topics
- **Purpose**: Removes completed topics
- **Condition**: `Status === "Pending"`
- **Output**: Filtered array

#### 4️⃣ Loop Over Topics
- **Purpose**: Sequential processing
- **Batch Size**: 1 topic per iteration
- **Output**: Single topic object

#### 5️⃣ Tavily Research
- **Purpose**: Web search for current information
- **API**: Tavily Search API
- **Parameters**:
  - Query: `{{ $json.Topic }}`
  - Max Results: 5
  - Search Depth: Advanced
- **Output**: Array of 5 research sources

#### 6️⃣ Split Research Results
- **Purpose**: Converts array to individual items
- **Field**: `results`
- **Output**: 5 separate items

#### 7️⃣ Aggregate Research
- **Purpose**: Combines research data
- **Method**: Aggregate all items
- **Output**: Single item with `researchContent` array

#### 8️⃣ Prepare Data
- **Purpose**: Formats data for AI models
- **Transformations**:
  ```javascript
  Topic: $('Loop Over Topics').first().json.Topic
  researchSummary: $json.data[0].content.replace(/\n/g, "\\n")
  ```
- **Output**: Clean topic + research summary

#### 9️⃣ Generate LinkedIn Post
- **Model**: Llama 3.1
- **Prompt**:
  ```
  You are a professional LinkedIn content writer.
  
  Write a thought-leadership post on: {{ Topic }}
  
  Context: {{ researchSummary }}
  
  Tone: Professional, insightful
  Length: 120-200 words
  Do not use hashtags.
  ```
- **Output**: `{ response: "..." }`

#### 🔟 Generate X Post
- **Model**: Llama 3.1
- **Prompt**:
  ```
  You are a social media copywriter.
  
  Write a concise, engaging tweet on: {{ Topic }}
  
  Context: {{ researchSummary }}
  
  Max 280 characters.
  Use hooks and clarity.
  No hashtags.
  ```
- **Output**: `{ response: "..." }`

#### 1️⃣1️⃣ Generate Blog Summary
- **Model**: Llama 3.1
- **Prompt**:
  ```
  You are a blog writer.
  
  Write a 150-200 word blog summary on: {{ Topic }}
  
  Use the following research: {{ researchSummary }}
  
  Tone: Informative, neutral, explanatory
  ```
- **Output**: `{ response: "..." }`

#### 1️⃣2️⃣ Merge
- **Purpose**: Combines all 3 AI outputs
- **Method**: Merge by position
- **Output**: Single item with all content

#### 1️⃣3️⃣ Format Update Data
- **Purpose**: Prepares data for sheet update
- **Fields**:
  ```javascript
  Topic: From Prepare Data
  LinkedIn_Post: From LinkedIn generator
  X_Post: From X generator
  Blog_Summary: From Blog generator
  Status: "Completed"
  Published_Date: Current timestamp
  ```

#### 1️⃣4️⃣ Update Sheet with Content
- **Purpose**: Writes data back to Google Sheets
- **Method**: Update by matching `Topic` column
- **Trigger**: Loops back to process next topic

---

## 🔑 API Keys & Credentials

### Tavily API Key

1. Sign up at [tavily.com](https://tavily.com)
2. Navigate to **Dashboard** → **API Keys**
3. Copy your API key
4. In n8n, update **Tavily Research** node:
   ```javascript
   Body Parameters → api_key: YOUR_TAVILY_API_KEY
   ```

### Google Sheets OAuth2

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project
3. Enable **Google Sheets API**
4. Create **OAuth 2.0 credentials**
5. Download JSON credentials
6. In n8n:
   - **Credentials** → **Create New** → **Google Sheets OAuth2 API**
   - Upload JSON file
   - Authorize access

---

## 🐛 Troubleshooting

### Common Issues

#### ❌ "JSON parameter needs to be valid JSON"

**Cause**: Malformed JSON in Ollama request body

**Solution**:
```javascript
// Ensure proper escaping
{
  "model": "llama3.1",
  "prompt": "Your prompt here with {{ $json.Topic }}",
  "stream": false
}
```

#### ❌ "Multiple matching items for item [0]"

**Cause**: Referencing wrong node or multiple items

**Solution**:
```javascript
// Use .first() for single item from previous node
Topic: {{ $('Loop Over Topics').first().json.Topic }}
```

#### ❌ "Filter doesn't work - all topics processed"

**Cause**: Filter node missing or misconfigured

**Solution**:
- Ensure **Filter Pending Topics** node exists
- Condition: `{{ $json.Status }} equals "Pending"`
- Check case sensitivity

#### ❌ "Ollama connection refused"

**Cause**: Ollama server not running

**Solution**:
```bash
# Start Ollama
ollama serve

# Verify
curl http://127.0.0.1:11434/api/tags
```

#### ❌ "Tavily API rate limit exceeded"

**Cause**: Too many requests

**Solution**:
- Reduce schedule frequency
- Decrease `max_results` in Tavily node
- Upgrade Tavily plan

---

## ⚡ Performance Optimization

### Speed Improvements

1. **Reduce Research Results**:
   ```javascript
   Tavily Node → max_results: 3 (instead of 5)
   ```

2. **Use Smaller Model**:
   ```bash
   ollama pull llama3.1:8b-instruct-q4_0
   ```

3. **Parallel Processing** (Advanced):
   - Modify to process 2-3 topics simultaneously
   - Requires understanding n8n's execution flow

### Resource Management

```bash
# Monitor Ollama memory usage
ollama ps

# Limit Ollama to specific GPU
CUDA_VISIBLE_DEVICES=0 ollama serve
```

---

## 🔒 Security Best Practices

### 1. API Key Protection

```javascript
// Use n8n credentials instead of hardcoding
// Never commit API keys to Git
```

### 2. Google Sheets Permissions

- Share sheet only with service account email
- Use read/write permissions (not owner)
- Enable 2FA on Google account

### 3. Local Network Security

```bash
# Bind Ollama to localhost only (default)
# Don't expose port 11434 publicly
```

### 4. Data Privacy

- Ollama runs locally - no data sent to external AI APIs
- Tavily API: Review their privacy policy
- Google Sheets: Enable encryption at rest

---

## ❓ FAQ

### Q: Can I use OpenAI/Claude instead of Ollama?

**A:** Yes! Replace HTTP Request nodes with OpenAI/Anthropic nodes:

```javascript
// OpenAI Configuration
Model: gpt-4-turbo-preview
API Key: YOUR_OPENAI_KEY

// Anthropic Configuration
Model: claude-sonnet-4
API Key: YOUR_ANTHROPIC_KEY
```

### Q: How do I add more platforms (e.g., Instagram)?

**A:** Duplicate one of the generation nodes:

1. Copy "Generate X Post" node
2. Rename to "Generate Instagram Post"
3. Modify prompt for Instagram format
4. Update Merge node to include 4 inputs
5. Add `Instagram_Post` column to Google Sheets

### Q: Can I run this on a schedule other than 6 hours?

**A:** Yes, edit Schedule Trigger node:

```javascript
// Every hour
Interval: 1 hour

// Daily at specific time
Cron: 0 9 * * *  // 9 AM daily

// Custom cron expression
Cron: 0 */3 * * *  // Every 3 hours
```

### Q: What if Tavily doesn't find relevant research?

**A:** The workflow will still generate content but with limited context. Consider:

- Improving topic phrasing
- Using fallback to Wikipedia/Google
- Adding error handling for empty results

### Q: How do I handle errors in production?

**A:** Add error handling nodes:

1. **Error Trigger**: Catches failed executions
2. **Send Email**: Notifies you of failures
3. **Webhook**: Sends alerts to Slack/Discord

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Reporting Bugs

1. Check existing issues first
2. Include n8n version, Ollama version, OS
3. Provide workflow JSON if possible
4. Include error logs

### Feature Requests

1. Describe the use case
2. Explain expected behavior
3. Suggest implementation approach

### Pull Requests

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Test thoroughly
4. Update README if needed
5. Submit PR with clear description

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 AHILL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨💻 Author

<div align="center">

### **AHILL**

[![GitHub](https://img.shields.io/badge/GitHub-AHILL--0121-181717?style=for-the-badge&logo=github)](https://github.com/AHILL-0121/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge&logo=google-chrome&logoColor=white)](https://github.com/AHILL-0121/)

**Automation Engineer | AI Enthusiast | Open Source Contributor**

*Building intelligent systems that scale*

</div>

---

## 🌟 Support

If you find this project helpful, please consider:

- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting features
- 🔗 Sharing with others

---

## 📞 Contact

- **GitHub Issues**: [Report a bug](https://github.com/AHILL-0121/Pinnacle_Projects/issues)
- **Email**: [MAIL](mailto:sa.education5211@gmail.com)

---

<div align="center">

**Made with ❤️ by [AHILL](https://github.com/AHILL-0121/)**

*Empowering creators with AI automation*

---

**Version 1.0.0** | Last Updated: February 2025

</div>
