# Website Audit LangGraph Agent

Standalone LangGraph agent for comprehensive website SEO auditing. This production-ready agent extracts HTML content from webpages, performs detailed SEO analysis, and provides actionable recommendations for improvement.

## Features

- **Comprehensive Content Extraction**: Automatically scrapes and extracts all SEO-relevant elements from any webpage
- **SEO Analysis**: Analyzes title tags, meta descriptions, heading structure, content quality, images, links, and technical SEO
- **Actionable Recommendations**: Provides prioritized action plans with immediate, short-term, and long-term improvements
- **Structured Reports**: Saves detailed audit reports in organized, easy-to-read text format

## Workflow

The agent follows a streamlined workflow:

1. **Extract Content** - Scrapes the webpage and extracts all SEO elements (title, meta tags, headings, images, links, structured data)
2. **Analyze SEO** - Performs comprehensive SEO analysis across multiple categories with scoring
3. **Generate Recommendations** - Creates a prioritized action plan with specific steps and expected impact

## Installation

Install dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Required Environment Variables

### Required: OpenAI API Key

Set your OpenAI API key using one of these methods:

**Option 1: Environment Variable**

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**Option 2: .env File**

Create a `.env` file in the same directory as the script:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The script will automatically load the `.env` file if `python-dotenv` is installed.

### Optional: LangSmith for LLM Logging and Observability

To view detailed LLM logs, traces, and monitor agent performance, add LangSmith environment variables to your `.env` file:

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

If LangSmith is not configured, the agent will work normally but without observability features.

## Usage

### Run as Standalone Script

```bash
python website_audit_agent.py
```

Or run as a module:

```bash
python -m website_audit_agent
```

### Customize the Example

Edit the `example_usage()` function in the script to customize:

- **URL**: The website URL to audit

Example:

```python
result = await agent.process(
    url="https://example.com"
)
```

### Use as a Library

You can also import and use the agent in your own Python code:

```python
from website_audit_agent import create_agent
import asyncio

async def audit_website():
    agent = create_agent()
    result = await agent.process(url="https://example.com")
    return result

# Run the audit
result = asyncio.run(audit_website())
print(f"Overall SEO Score: {result['overall_score']}/100")
```

## Output

Results are automatically saved to the `output` folder in the same directory as the script:

### File Structure

- **`audit_{domain}_{timestamp}.md`** - Comprehensive SEO audit report in markdown format with scores, analysis, and recommendations

### Output Location

- When run as a script: `{script_directory}/output/`
- When run as a module: `{current_working_directory}/output/`

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

3. **Action Plan**
   - Immediate Actions (do first)
   - Short-term Improvements (1-4 weeks)
   - Long-term Strategy (1-6 months)
   - Expected improvements and timeline

## SEO Elements Analyzed

The agent performs comprehensive analysis of:

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
- **Structured Data**: Schema markup
- **Open Graph Tags**: Social media optimization

### Scoring System

Each category is scored from 0-100:
- **90-100**: Excellent
- **80-89**: Good
- **70-79**: Fair
- **60-69**: Needs Improvement
- **Below 60**: Poor

## Dependencies

The agent requires:

- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integration
- `langgraph` - Graph-based agent orchestration
- `aiohttp` - Async HTTP client for web scraping
- `beautifulsoup4` - HTML parsing
- `python-dotenv` - Environment variable loading (optional)

All dependencies are listed in `requirements.txt`.

## Logging

The agent provides detailed logging during execution:

- Node execution status
- Content extraction progress
- SEO analysis progress
- Recommendation generation
- Workflow completion status

All logs are printed to the console with clear visual indicators.

## Error Handling

The agent includes comprehensive error handling:

- Invalid URL validation
- Content extraction errors
- SEO analysis failures
- Recommendation generation errors
- File writing errors

Errors are logged and reported clearly, allowing for easy debugging.

## Example Output

```
🚀 Starting Website Audit Agent Workflow
================================================================================

🔍 NODE: extract_content_node - Starting content extraction
================================================================================
✅ Content extracted successfully (Word count: 1,234)

📊 NODE: analyze_seo_node - Starting SEO analysis
================================================================================
✅ SEO analysis completed (Overall score: 78/100)

💡 NODE: generate_recommendations_node - Generating recommendations
================================================================================
✅ Recommendations generated successfully

✅ Workflow completed successfully!
================================================================================

✅ Audit report saved to: output/audit_example-com_20231215_143022.txt
```

## Use Cases

- **SEO Audits**: Comprehensive analysis of website SEO health
- **Competitor Analysis**: Compare SEO implementations across websites
- **Pre-Launch Checks**: Ensure new websites are SEO-optimized before launch
- **Regular Monitoring**: Track SEO improvements over time
- **Client Reports**: Generate professional SEO audit reports for clients

## Limitations

- Requires JavaScript-free HTML content (some dynamic content may not be captured)
- Does not check page load speed or Core Web Vitals
- Does not analyze backlinks or domain authority
- Does not check mobile responsiveness (HTML only)

## Contributing

Feel free to extend the agent with additional SEO checks:
- Page speed analysis
- Mobile-friendliness
- Security headers
- Accessibility audit
- Schema validation

## License

This is a standalone agent template. Use and modify as needed for your projects.

# website-audit-agent
