# Website SEO Audit AI Agent

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.18-green.svg)
![AI Agent](https://img.shields.io/badge/AI-Agent-FF6B6B.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)

Production-ready **FastAPI** application powered by an **AI Agent** for comprehensive website SEO and performance auditing. This REST API uses a LangGraph-based AI agent to extract HTML content, perform detailed SEO analysis, measure page speed, and generate professional PDF reports with actionable recommendations.

**An intelligent AI agent that automates website SEO auditing - also available as a standalone Python script for one-off audits.**

## Features

This AI agent provides:

- **Comprehensive Content Extraction**: The AI agent automatically scrapes and extracts all SEO-relevant elements from any webpage
- **AI-Powered SEO Analysis**: The AI agent analyzes title tags, meta descriptions, heading structure, content quality, images, links, and technical SEO with intelligent scoring
- **Page Speed Analysis**: The AI agent measures page load time (TTFB and total), response size, compression, caching, CDN usage, and HTTP version
- **Actionable Recommendations**: The AI agent provides prioritized action plans with immediate, short-term, and long-term improvements
- **Professional PDF Reports**: The AI agent generates detailed audit reports in PDF format with scores, analysis, speed metrics, and recommendations
- **REST API**: Production-ready FastAPI endpoint powered by the AI agent with authentication, rate limiting, and CORS support
- **Standalone AI Agent**: Can be run directly as a Python script for one-off audits

## Architecture

This is a **FastAPI application** powered by an **AI Agent** with the following components:

1. **`main.py`** - FastAPI REST API server (main entry point)
2. **`website_audit_agent.py`** - Core LangGraph AI agent with workflow orchestration
3. **`pdf_generator.py`** - PDF report generation using WeasyPrint

The FastAPI server (`main.py`) is the primary interface, providing REST endpoints for SEO auditing. The AI agent can also be run standalone for direct Python usage.

## Workflow

The AI agent follows a streamlined 4-step workflow:

1. **Extract Content** - The AI agent scrapes the webpage and extracts all SEO elements (title, meta tags, headings, images, links, structured data)
2. **Analyze SEO** - The AI agent performs comprehensive SEO analysis across multiple categories with AI-powered scoring
3. **Check Page Speed** - The AI agent measures page load time, response metrics, and performance optimizations (compression, caching, CDN)
4. **Generate Recommendations** - The AI agent creates a prioritized action plan with specific steps and expected impact

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key

### Quick Setup

1. Clone or download the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables (see below)

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

### Required: OpenAI API Key

Set your OpenAI API key using one of these methods:

**Option 1: Environment Variable**

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**Option 2: .env File**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The script will automatically load the `.env` file if `python-dotenv` is installed.

### Optional: OpenAI Model Configuration

You can configure which OpenAI model the AI agent uses by setting the `OPENAI_MODEL` environment variable:

**Option 1: Environment Variable**

```bash
export OPENAI_MODEL=gpt-4o-mini
```

**Option 2: .env File**

```env
OPENAI_MODEL=gpt-4o-mini
```

**Supported Models:**
- `gpt-4o-2024-11-20` (default) - Latest GPT-4o model with best performance
- `gpt-4o` - GPT-4o model
- `gpt-4o-mini` - Faster and cheaper GPT-4o variant
- `gpt-4-turbo` - GPT-4 Turbo model
- `gpt-3.5-turbo` - GPT-3.5 Turbo (faster, less accurate)

**Note:** If not set, the AI agent defaults to `gpt-4o-2024-11-20` for optimal results.

### Optional: LangSmith for LLM Logging and Observability

To view detailed LLM logs, traces, and monitor agent performance:

```env
# LangSmith Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=website-audit-agent
```

**Getting Your LangSmith API Key:**

1. Sign up for a free account at [https://smith.langchain.com](https://smith.langchain.com)
2. Navigate to Settings → API Keys
3. Create a new API key
4. Copy the key to your `.env` file

**Benefits of LangSmith:**

- View all LLM requests and responses in real-time
- Monitor token usage and costs
- Debug agent workflow execution
- Track performance metrics
- Analyze prompt effectiveness

### API Server Configuration (For FastAPI Mode)

When running the API server, you can configure these optional environment variables:

```env
# API Server Configuration
API_KEY=your_api_key_here                    # API key for authentication (optional)
RATE_LIMIT_PER_MINUTE=10                      # Rate limit per minute (default: 10)
PORT=9000                                      # Server port (default: 9000)
HOST=0.0.0.0                                  # Server host (default: 0.0.0.0)
CORS_ORIGINS=*                                 # CORS allowed origins (default: *)
DEBUG=False                                    # Enable debug mode (default: False)

# AI Agent Configuration
OPENAI_MODEL=gpt-4o-2024-11-20                # OpenAI model to use (default: gpt-4o-2024-11-20)
```

**Note:** If `API_KEY` is not set, the API will be accessible without authentication (not recommended for production).

## Usage

### Primary: FastAPI REST API

This is a **FastAPI application**. Start the API server to access the REST endpoints.

#### Start the API Server

**Option 1: Using the startup script (Recommended)**

```bash
chmod +x start_api.sh
./start_api.sh
```

**Option 2: Direct Python execution**

```bash
python main.py
```

**Option 3: Using uvicorn directly**

```bash
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

The API will be available at `http://localhost:9000`

**Interactive API Documentation:**
- Swagger UI: `http://localhost:9000/docs`
- ReDoc: `http://localhost:9000/redoc`

#### API Endpoints

**1. Health Check**

```bash
GET /health
```

Returns API health status and version.

**Response:**
```json
{
  "status": "healthy",
  "message": "Website SEO Audit API is running",
  "version": "1.0.0"
}
```

**2. Root Endpoint**

```bash
GET /
```

Returns API information and available endpoints.

**3. Perform SEO Audit**

```bash
POST /api/v1/audit
```

**Headers:**
```
X-API-Key: your_api_key_here  # Required if API_KEY is set
Content-Type: application/json
```

**Request Body:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "status": "success",
  "url": "https://example.com",
  "extracted_content": { ... },
  "seo_audit": { ... },
  "speed_audit": { ... },
  "recommendations": { ... },
  "overall_score": 78,
  "processing_status": "completed",
  "pdf_path": "/path/to/audit_example-com_20231215_143022.pdf",
  "pdf_filename": "audit_example-com_20231215_143022.pdf"
}
```

**Example using cURL:**

```bash
curl -X POST "http://localhost:9000/api/v1/audit" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Example using Python:**

```python
import requests

url = "http://localhost:9000/api/v1/audit"
headers = {
    "X-API-Key": "your_api_key_here",
    "Content-Type": "application/json"
}
data = {
    "url": "https://example.com"
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(f"Overall SEO Score: {result['overall_score']}/100")
print(f"PDF Path: {result['pdf_path']}")
```

**Example using JavaScript (fetch):**

```javascript
const response = await fetch('http://localhost:9000/api/v1/audit', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your_api_key_here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    url: 'https://example.com'
  })
});

const result = await response.json();
console.log(`Overall SEO Score: ${result.overall_score}/100`);
```

### Alternative: Standalone AI Agent Script

For one-off audits without running the API server, you can run the AI agent directly:

```bash
python website_audit_agent.py
```

Or as a module:

```bash
python -m website_audit_agent
```

**Customizing the URL:**

Edit the `example_usage()` function in `website_audit_agent.py`:

```python
url = "https://example.com"  # Change this URL
```

### Alternative: Use AI Agent as a Library

You can also import and use the AI agent in your own Python code:

```python
from website_audit_agent import create_agent
import asyncio

async def audit_website():
    # Create AI agent instance
    agent = create_agent()
    result = await agent.process(url="https://example.com")
    return result

# Run the audit using the AI agent
result = asyncio.run(audit_website())
print(f"Overall SEO Score: {result['overall_score']}/100")
print(f"Speed Score: {result['speed_audit']['overall_speed_score']}/100")
```

## Output

### File Structure

Results are automatically saved to the `output` folder:

- **`audit_{domain}_{timestamp}.pdf`** - Comprehensive SEO audit report in PDF format

### Output Location

- **FastAPI mode** (default): `{api_directory}/output/`
- **Standalone script**: `{script_directory}/output/`
- **Module execution**: `{current_working_directory}/output/`

The `output` folder is automatically created if it doesn't exist.

### Report Contents

**Audit Report includes:**

1. **Page Information**
   - Title and length
   - Meta description and length
   - Word count
   - Heading structure (H1-H6)
   - Image statistics (with/without alt text)
   - Link statistics (internal/external)

2. **SEO Audit Results**
   - Title Analysis (score, issues, suggestions)
   - Meta Description Analysis
   - Heading Structure Analysis
   - Content Analysis
   - Image Optimization
   - Link Structure
   - Technical SEO
   - Priority Fixes with solutions
   - Strengths
   - Quick Wins

3. **Page Speed Analysis**
   - Overall Speed Score (0-100)
   - Response Time (seconds and milliseconds)
   - Time To First Byte (TTFB)
   - Page Size (MB and bytes)
   - HTTP Version and Status Code
   - Performance Optimizations:
     - Compression status (GZIP/Brotli)
     - Cache headers presence
     - CDN detection
   - Speed Issues and Optimization Suggestions

4. **Action Plan**
   - Immediate Actions (do first)
   - Short-term Improvements (1-4 weeks)
   - Long-term Strategy (1-6 months)
   - Expected improvements and timeline

## SEO Elements Analyzed

The AI agent performs comprehensive analysis of:

### On-Page SEO
- **Title Tag**: Length, keyword usage, uniqueness
- **Meta Description**: Length, call-to-action, keyword usage
- **Headings**: H1 count, hierarchy, keyword usage
- **Content**: Word count, readability, keyword density
- **Images**: Alt text usage, file names, optimization
- **Internal Links**: Link structure, anchor text
- **External Links**: Quality, relevance

### Technical SEO
- **Canonical URLs**: Proper implementation
- **Robots Meta Tags**: Indexing directives
- **Structured Data**: Schema markup (JSON-LD)
- **Open Graph Tags**: Social media optimization

### Page Speed Analysis
- **Response Time**: Total page load time measurement (target: <2 seconds)
- **Time To First Byte (TTFB)**: Server response time
- **Page Size**: Total response size (target: <2 MB)
- **Compression**: GZIP/Brotli compression detection
- **Caching**: Cache-Control and Expires headers
- **CDN Usage**: Content Delivery Network detection
- **HTTP Version**: HTTP/1.1 or HTTP/2 detection
- **Performance Score**: Overall speed score (0-100) based on response time, size, and optimizations

**Speed Scoring Breakdown:**
- Response time < 0.5s: 100 points
- Response time < 1.0s: 95 points
- Response time < 1.5s: 85 points
- Response time < 2.0s: 75 points
- Response time < 3.0s: 60 points
- Response time < 5.0s: 40 points
- Response time < 10.0s: 20 points
- Response time >= 10.0s: 10 points

**Size Scoring:**
- Page size < 0.5 MB: 100 points
- Page size < 1.0 MB: 95 points
- Page size < 2.0 MB: 80 points
- Page size < 3.0 MB: 60 points
- Page size < 5.0 MB: 40 points
- Page size < 10.0 MB: 20 points
- Page size >= 10.0 MB: 10 points

**Overall Speed Score:** Weighted average (75% response time, 25% page size) with optimization bonuses.

### Scoring System

Each category is scored from 0-100:
- **90-100**: Excellent
- **80-89**: Good
- **70-79**: Fair
- **60-69**: Needs Improvement
- **Below 60**: Poor

## Security Features (API Mode)

The FastAPI server includes production-ready security features:

- **API Key Authentication**: Optional API key protection via `X-API-Key` header
- **Rate Limiting**: Configurable rate limiting per IP address (default: 10 requests/minute)
- **CORS Configuration**: Configurable Cross-Origin Resource Sharing
- **Input Validation**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes
- **Logging**: Detailed logging for debugging and monitoring

## Dependencies

### Core Dependencies

- `langchain==0.3.3` - Core LangChain framework
- `langchain-openai==0.2.0` - OpenAI integration
- `langchain-community==0.3.2` - Community integrations
- `langgraph==0.2.18` - Graph-based agent orchestration
- `aiohttp==3.9.1` - Async HTTP client for web scraping
- `beautifulsoup4==4.12.3` - HTML parsing

### PDF Generation

- `weasyprint>=60.0` - PDF generation from HTML
- `jinja2>=3.1.0` - Template engine
- `markdown>=3.5.0` - Markdown to HTML conversion
- `bleach>=6.0.0` - HTML sanitization

### API Server

- `fastapi==0.115.0` - Modern web framework
- `uvicorn[standard]==0.32.0` - ASGI server
- `slowapi==0.1.9` - Rate limiting
- `gunicorn==21.2.0` - Production WSGI server

### Configuration

- `pydantic==2.10.4` - Data validation
- `pydantic-settings==2.7.0` - Settings management
- `python-dotenv==1.0.0` - Environment variable loading

All dependencies are listed in `requirements.txt`.

## Logging

The AI agent provides detailed logging during execution:

- Node execution status
- Content extraction progress
- SEO analysis progress
- Speed check progress
- Recommendation generation
- Workflow completion status
- API request/response logging

All logs are printed to the console with clear visual indicators and timestamps.

## Error Handling

The AI agent includes comprehensive error handling:

- Invalid URL validation
- Content extraction errors
- SEO analysis failures
- Speed check timeouts
- Recommendation generation errors
- PDF generation errors (non-critical, falls back gracefully)
- File writing errors
- API authentication errors
- Rate limit exceeded errors

Errors are logged and reported clearly, allowing for easy debugging.

## Example Output

### API Response Example

```json
{
  "status": "success",
  "url": "https://example.com",
  "overall_score": 78,
  "processing_status": "completed",
  "pdf_filename": "audit_example-com_20231215_143022.pdf",
  "pdf_path": "/path/to/output/audit_example-com_20231215_143022.pdf",
  "extracted_content": {
    "title": "Example Page",
    "word_count": 1234,
    ...
  },
  "seo_audit": {
    "overall_seo_score": 78,
    "title_analysis": { ... },
    ...
  },
  "speed_audit": {
    "overall_speed_score": 85,
    "response_time_seconds": 1.234,
    ...
  },
  "recommendations": {
    "immediate_actions": [ ... ],
    ...
  }
}
```

### Standalone Script Output

When running the standalone script, you'll see console output:

```
🚀 Starting Website Audit Agent Workflow
================================================================================

🔍 NODE: extract_content_node - Starting content extraction
================================================================================
✅ Content extracted successfully (Word count: 1,234)

📊 NODE: analyze_seo_node - Starting SEO analysis
================================================================================
✅ SEO analysis completed (Overall score: 78/100)

⚡ NODE: check_speed_node - Checking page speed
================================================================================
✅ Speed check completed (Response time: 1.234s, Score: 85/100)

💡 NODE: generate_recommendations_node - Generating recommendations
================================================================================
✅ Recommendations generated successfully

✅ Workflow completed successfully!
================================================================================

✅ Audit report saved to: output/audit_example-com_20231215_143022.pdf
   Full path: /path/to/output/audit_example-com_20231215_143022.pdf
```

## Use Cases

This AI agent is perfect for:

- **SEO Audits**: Comprehensive analysis of website SEO health using AI-powered insights
- **Competitor Analysis**: Compare SEO implementations across websites with automated AI agent analysis
- **Pre-Launch Checks**: Ensure new websites are SEO-optimized before launch with AI agent validation
- **Regular Monitoring**: Track SEO improvements over time with scheduled AI agent audits
- **Client Reports**: Generate professional SEO audit reports for clients using the AI agent
- **API Integration**: Integrate AI agent SEO auditing into your own applications
- **Batch Processing**: Process multiple URLs programmatically with the AI agent
- **Performance Monitoring**: Track page speed improvements with AI agent metrics

## Limitations

- Requires JavaScript-free HTML content (some dynamic content may not be captured)
- Speed check measures server response time and total download time (not full page load with JavaScript execution)
- Does not analyze backlinks or domain authority
- Does not check mobile responsiveness (HTML only)
- Does not measure Core Web Vitals (LCP, FID, CLS) - only server response metrics
- Rate limited to prevent abuse (configurable in API mode)

## Production Deployment

### Using Gunicorn

For production deployment, use Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9000
```

### Environment Variables for Production

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# API Security
API_KEY=your_secure_api_key_here
RATE_LIMIT_PER_MINUTE=20

# Server Configuration
PORT=9000
HOST=0.0.0.0
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
DEBUG=False

# AI Agent Configuration
OPENAI_MODEL=gpt-4o-2024-11-20  # or gpt-4o-mini for cost savings
```

### Docker Deployment (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9000"]
```

## Contributing

Feel free to extend the AI agent with additional SEO checks:

- Core Web Vitals (LCP, FID, CLS)
- Mobile-friendliness testing
- Security headers analysis
- Accessibility audit (WCAG compliance)
- Schema validation
- Lighthouse integration
- Multi-page site crawling
- Backlink analysis integration

## Troubleshooting

### Common Issues

**1. PDF Generation Fails**
- Ensure WeasyPrint dependencies are installed (may require system libraries on Linux)
- Check that `pdf_generator.py` is accessible
- The agent will continue without PDF if generation fails

**2. Speed Check Always Returns 100**
- This was fixed in a recent update - ensure you're using the latest version of the AI agent
- Speed now measures total download time, not just headers

**3. API Authentication Not Working**
- Check that `API_KEY` environment variable is set
- Verify the `X-API-Key` header is included in requests
- If `API_KEY` is not set, API is accessible without authentication

**4. Rate Limit Errors**
- Adjust `RATE_LIMIT_PER_MINUTE` environment variable
- Use different IP addresses for testing
- Implement request queuing for batch processing

## License

This is a standalone agent template. Use and modify as needed for your projects.

## Support

For issues, questions, or contributions, please refer to the project repository or create an issue.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.3-1C3C3C?logo=langchain)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.18-FF6B6B?logo=graphql)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai)
![WeasyPrint](https://img.shields.io/badge/WeasyPrint-PDF-FF5733)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-499848?logo=uvicorn)

---

**Version:** 1.0.0  
**Last Updated:** December 2025
