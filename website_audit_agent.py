"""
Standalone LangGraph Agent for Website SEO Audit

This is a production-ready, standalone backend agent that uses LangGraph
to orchestrate website SEO audit workflows with integrated browser tools.

INSTALLATION:
    pip install -r requirements.txt

REQUIRED ENVIRONMENT VARIABLES:
    - OPENAI_API_KEY: Your OpenAI API key (required)

    You can set it in:
    1. Environment variable: export OPENAI_API_KEY=your_key
    2. .env file in the same directory as this script

USAGE:
    python website_audit_agent.py

    Or as a module:
    python -m website_audit_agent

OUTPUT:
    Results are saved to the 'output' folder:
    - audit_[domain]_[timestamp].md: Detailed SEO audit report in markdown format with scores and recommendations
"""

import logging
import json
import re
import time
import os
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Load .env file from the same directory as this script (if running as script)
    # or from current working directory
    try:
        script_dir = Path(__file__).parent.absolute()
        env_path = script_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Try current working directory
            load_dotenv()
    except NameError:
        # __file__ not available when running as module, try current directory
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables only

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment (will be checked when LLMService is initialized)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================================
# State Definition
# ============================================================================


class WebsiteAuditState(TypedDict):
    """State for the website audit agent."""

    messages: Annotated[List, lambda x, y: x + y]
    url: str
    extracted_content: Optional[Dict[str, Any]]
    html_structure: Optional[Dict[str, Any]]
    seo_audit: Optional[Dict[str, Any]]
    technical_audit: Optional[Dict[str, Any]]
    content_audit: Optional[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]]
    overall_score: Optional[int]
    status: str
    error: Optional[str]


# ============================================================================
# Browser Tools
# ============================================================================


@tool
async def extract_webpage_content(url: str) -> str:
    """
    Extract HTML content and SEO elements from a webpage for auditing.

    Args:
        url: The URL to extract content from

    Returns:
        JSON string containing extracted content with title, content, meta tags, headings, images, links, and structure
    """
    try:
        logger.info(f"Extracting webpage content from URL: {url}")

        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return json.dumps({"error": "Invalid URL format"})

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    return json.dumps(
                        {
                            "error": f"Failed to fetch URL, status code: {response.status}"
                        }
                    )

                html_content = await response.text()

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()

        # Extract meta tags
        meta_description = ""
        meta_keywords = ""
        og_tags = {}

        meta_desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if meta_desc_tag and meta_desc_tag.get("content"):
            meta_description = meta_desc_tag["content"].strip()

        meta_keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords_tag and meta_keywords_tag.get("content"):
            meta_keywords = meta_keywords_tag["content"].strip()

        # Extract Open Graph tags
        for og_tag in soup.find_all("meta", property=re.compile(r"^og:")):
            property_name = og_tag.get("property", "")
            content = og_tag.get("content", "")
            og_tags[property_name] = content

        # Extract headings structure
        headings = {
            f"h{i}": [h.get_text().strip() for h in soup.find_all(f"h{i}")]
            for i in range(1, 7)
        }

        # Extract images with alt text analysis
        images = []
        for img in soup.find_all("img"):
            images.append(
                {
                    "src": img.get("src", ""),
                    "alt": img.get("alt", ""),
                    "has_alt": bool(img.get("alt")),
                }
            )

        # Extract links (internal and external)
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            is_internal = href.startswith("/") or parsed_url.netloc in href
            links.append(
                {
                    "href": href,
                    "text": link.get_text().strip(),
                    "is_internal": is_internal,
                }
            )

        # Extract main content
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
            or soup.find("div", id="content")
        )

        if main_content:
            paragraphs = main_content.find_all(["p"])
        else:
            paragraphs = soup.find_all(["p"])

        body_text = "\n\n".join(
            [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
        )

        # Check for structured data (JSON-LD)
        structured_data = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                structured_data.append(json.loads(script.string))
            except:
                pass

        # Extract canonical URL
        canonical = ""
        canonical_tag = soup.find("link", rel="canonical")
        if canonical_tag:
            canonical = canonical_tag.get("href", "")

        # Check for robots meta tag
        robots_meta = ""
        robots_tag = soup.find("meta", attrs={"name": "robots"})
        if robots_tag:
            robots_meta = robots_tag.get("content", "")

        result = {
            "url": url,
            "title": title,
            "title_length": len(title),
            "meta_description": meta_description,
            "meta_description_length": len(meta_description),
            "meta_keywords": meta_keywords,
            "og_tags": og_tags,
            "canonical_url": canonical,
            "robots_meta": robots_meta,
            "headings": headings,
            "content": body_text,
            "word_count": len(body_text.split()),
            "images": {
                "total": len(images),
                "with_alt": len([img for img in images if img["has_alt"]]),
                "without_alt": len([img for img in images if not img["has_alt"]]),
                "details": images[:20],  # Limit to first 20 for analysis
            },
            "links": {
                "total": len(links),
                "internal": len([l for l in links if l["is_internal"]]),
                "external": len([l for l in links if not l["is_internal"]]),
            },
            "structured_data": structured_data,
            "has_h1": len(headings.get("h1", [])) > 0,
            "h1_count": len(headings.get("h1", [])),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error extracting content from URL: {str(e)}")
        return json.dumps({"error": f"Failed to extract content: {str(e)}"})


# ============================================================================
# LLM Service Functions
# ============================================================================


class LLMService:
    """Service for interacting with the LLM."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is not set. Please set OPENAI_API_KEY environment variable or pass it as a parameter."
            )

        self.llm = ChatOpenAI(
            model="gpt-4o-2024-11-20",
            temperature=0.3,
            streaming=False,
            verbose=False,
            api_key=self.api_key,
        )

    async def analyze_seo(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SEO elements of the webpage."""
        try:
            logger.info("Starting SEO analysis")

            from langchain.schema import StrOutputParser

            template = """You are an expert SEO auditor. Analyze the following webpage data and provide a comprehensive SEO audit.

Webpage Data:
- URL: {url}
- Title: {title} (Length: {title_length} characters)
- Meta Description: {meta_description} (Length: {meta_description_length} characters)
- H1 Count: {h1_count}
- H1 Tags: {h1_tags}
- H2 Count: {h2_count}
- Word Count: {word_count}
- Images: {total_images} total, {images_with_alt} with alt text, {images_without_alt} without alt text
- Links: {total_links} total, {internal_links} internal, {external_links} external
- Has Canonical URL: {has_canonical}
- Robots Meta: {robots_meta}
- Structured Data: {has_structured_data}

Provide your analysis in JSON format:

```json
{{
    "title_analysis": {{
        "score": 85,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "meta_description_analysis": {{
        "score": 90,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "heading_structure": {{
        "score": 75,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "content_analysis": {{
        "score": 80,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "image_optimization": {{
        "score": 70,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "link_structure": {{
        "score": 85,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "technical_seo": {{
        "score": 90,
        "issues": ["List any issues"],
        "suggestions": ["List suggestions"]
    }},
    "overall_seo_score": 82,
    "priority_fixes": [
        {{
            "issue": "Issue description",
            "priority": "high|medium|low",
            "impact": "Impact description",
            "solution": "How to fix it"
        }}
    ],
    "strengths": ["List strengths"],
    "quick_wins": ["Easy improvements with high impact"]
}}
```"""

            from langchain.prompts import PromptTemplate

            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "url",
                    "title",
                    "title_length",
                    "meta_description",
                    "meta_description_length",
                    "h1_count",
                    "h1_tags",
                    "h2_count",
                    "word_count",
                    "total_images",
                    "images_with_alt",
                    "images_without_alt",
                    "total_links",
                    "internal_links",
                    "external_links",
                    "has_canonical",
                    "robots_meta",
                    "has_structured_data",
                ],
            )

            chain = prompt | self.llm | StrOutputParser()

            h1_tags = ", ".join(content.get("headings", {}).get("h1", [])) or "None"
            h2_count = len(content.get("headings", {}).get("h2", []))

            output_text = await chain.ainvoke(
                {
                    "url": content.get("url", ""),
                    "title": content.get("title", "No title"),
                    "title_length": content.get("title_length", 0),
                    "meta_description": content.get(
                        "meta_description", "No meta description"
                    ),
                    "meta_description_length": content.get(
                        "meta_description_length", 0
                    ),
                    "h1_count": content.get("h1_count", 0),
                    "h1_tags": h1_tags,
                    "h2_count": h2_count,
                    "word_count": content.get("word_count", 0),
                    "total_images": content.get("images", {}).get("total", 0),
                    "images_with_alt": content.get("images", {}).get("with_alt", 0),
                    "images_without_alt": content.get("images", {}).get(
                        "without_alt", 0
                    ),
                    "total_links": content.get("links", {}).get("total", 0),
                    "internal_links": content.get("links", {}).get("internal", 0),
                    "external_links": content.get("links", {}).get("external", 0),
                    "has_canonical": "Yes" if content.get("canonical_url") else "No",
                    "robots_meta": content.get("robots_meta", "Not set"),
                    "has_structured_data": (
                        "Yes" if content.get("structured_data") else "No"
                    ),
                }
            )

            # Parse output
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", output_text)
            if json_match:
                seo_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r"\{[\s\S]*\}", output_text)
                if json_match:
                    seo_data = json.loads(json_match.group(0))
                else:
                    raise ValueError("Failed to parse SEO analysis JSON")

            return seo_data

        except Exception as e:
            logger.error(f"Error in SEO analysis: {str(e)}")
            raise ValueError(f"SEO analysis failed: {str(e)}")

    async def generate_recommendations(
        self, content: Dict[str, Any], seo_audit: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable recommendations based on audit results."""
        try:
            logger.info("Generating recommendations")

            from langchain.schema import StrOutputParser

            recommend_prompt = ChatPromptTemplate.from_template(
                """You are an SEO consultant providing actionable recommendations. Based on the audit results, create a prioritized action plan.

Audit Summary:
- Overall SEO Score: {overall_score}/100
- Title Score: {title_score}/100
- Meta Description Score: {meta_score}/100
- Content Score: {content_score}/100
- Technical SEO Score: {technical_score}/100

Priority Fixes:
{priority_fixes}

Create a comprehensive action plan in JSON format:

```json
{{
    "immediate_actions": [
        {{
            "action": "Action description",
            "expected_impact": "high|medium|low",
            "implementation_time": "Time estimate",
            "steps": ["Step 1", "Step 2"]
        }}
    ],
    "short_term_improvements": [
        {{
            "action": "Action description",
            "expected_impact": "high|medium|low",
            "implementation_time": "Time estimate",
            "steps": ["Step 1", "Step 2"]
        }}
    ],
    "long_term_strategy": [
        {{
            "action": "Action description",
            "expected_impact": "high|medium|low",
            "implementation_time": "Time estimate",
            "steps": ["Step 1", "Step 2"]
        }}
    ],
    "estimated_overall_improvement": "X points increase in SEO score",
    "timeline": "Estimated time to complete all improvements"
}}
```"""
            )

            chain = recommend_prompt | self.llm | StrOutputParser()

            priority_fixes_str = "\n".join(
                [
                    f"- {fix.get('issue', '')} (Priority: {fix.get('priority', 'medium')})"
                    for fix in seo_audit.get("priority_fixes", [])
                ]
            )

            output_text = await chain.ainvoke(
                {
                    "overall_score": seo_audit.get("overall_seo_score", 0),
                    "title_score": seo_audit.get("title_analysis", {}).get("score", 0),
                    "meta_score": seo_audit.get("meta_description_analysis", {}).get(
                        "score", 0
                    ),
                    "content_score": seo_audit.get("content_analysis", {}).get(
                        "score", 0
                    ),
                    "technical_score": seo_audit.get("technical_seo", {}).get(
                        "score", 0
                    ),
                    "priority_fixes": priority_fixes_str,
                }
            )

            # Parse output
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", output_text)
            if json_match:
                recommendations = json.loads(json_match.group(1))
            else:
                json_match = re.search(r"\{[\s\S]*\}", output_text)
                if json_match:
                    recommendations = json.loads(json_match.group(0))
                else:
                    raise ValueError("Failed to parse recommendations JSON")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise ValueError(f"Recommendation generation failed: {str(e)}")


# ============================================================================
# Graph Nodes
# ============================================================================

llm_service = LLMService()


async def extract_content_node(state: WebsiteAuditState) -> WebsiteAuditState:
    """Extract content from the website URL."""
    try:
        print("\n" + "=" * 80)
        print("🔍 NODE: extract_content_node - Starting content extraction")
        print("=" * 80)
        logger.info(f"Extracting content from URL: {state['url']}")

        content_json = await extract_webpage_content.ainvoke(state["url"])
        content_data = json.loads(content_json)

        if "error" in content_data:
            print(f"❌ Error extracting content: {content_data['error']}")
            return {**state, "status": "error", "error": content_data["error"]}

        print(
            f"✅ Content extracted successfully (Word count: {content_data.get('word_count', 0)})"
        )
        return {
            **state,
            "extracted_content": content_data,
            "status": "content_extracted",
        }
    except Exception as e:
        print(f"❌ Error in extract_content_node: {str(e)}")
        logger.error(f"Error in extract_content_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def analyze_seo_node(state: WebsiteAuditState) -> WebsiteAuditState:
    """Analyze SEO elements of the webpage."""
    try:
        print("\n" + "=" * 80)
        print("📊 NODE: analyze_seo_node - Starting SEO analysis")
        print("=" * 80)
        logger.info("Analyzing SEO")

        content = state.get("extracted_content", {})

        if not content:
            return {**state, "status": "error", "error": "No content to analyze"}

        seo_audit = await llm_service.analyze_seo(content)

        print(
            f"✅ SEO analysis completed (Overall score: {seo_audit.get('overall_seo_score', 0)}/100)"
        )

        return {
            **state,
            "seo_audit": seo_audit,
            "overall_score": seo_audit.get("overall_seo_score", 0),
            "status": "seo_analyzed",
        }
    except Exception as e:
        print(f"❌ Error in analyze_seo_node: {str(e)}")
        logger.error(f"Error in analyze_seo_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def generate_recommendations_node(state: WebsiteAuditState) -> WebsiteAuditState:
    """Generate actionable recommendations."""
    try:
        print("\n" + "=" * 80)
        print("💡 NODE: generate_recommendations_node - Generating recommendations")
        print("=" * 80)
        logger.info("Generating recommendations")

        content = state.get("extracted_content", {})
        seo_audit = state.get("seo_audit", {})

        if not seo_audit:
            return state

        recommendations = await llm_service.generate_recommendations(content, seo_audit)

        print(f"✅ Recommendations generated successfully")

        return {
            **state,
            "recommendations": recommendations,
            "status": "recommendations_generated",
        }
    except Exception as e:
        print(f"❌ Error in generate_recommendations_node: {str(e)}")
        logger.error(f"Error in generate_recommendations_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def agent_node(state: WebsiteAuditState) -> WebsiteAuditState:
    """Main agent node that orchestrates the workflow."""
    try:
        # Determine next step based on current status
        current_status = state.get("status", "initialized")

        if current_status == "initialized":
            # Start by extracting content
            return await extract_content_node(state)

        elif current_status == "content_extracted":
            # Move to SEO analysis
            return await analyze_seo_node(state)

        elif current_status == "seo_analyzed":
            # Move to recommendations
            return await generate_recommendations_node(state)

        elif current_status == "recommendations_generated":
            # Workflow complete
            return {**state, "status": "completed"}

        return state

    except Exception as e:
        logger.error(f"Error in agent_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


def should_continue(state: WebsiteAuditState) -> Literal["continue", "end"]:
    """Determine if the workflow should continue or end."""
    status = state.get("status", "")

    if status == "completed":
        return "end"
    elif status == "error":
        return "end"
    else:
        return "continue"


# ============================================================================
# Graph Construction
# ============================================================================


def create_website_audit_agent():
    """Create and compile the website audit LangGraph agent."""

    # Create the graph
    workflow = StateGraph(WebsiteAuditState)

    # Add nodes
    workflow.add_node("agent", agent_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "agent",
            "end": END,
        },
    )

    # Compile the graph with memory
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


# ============================================================================
# Agent Interface
# ============================================================================


class WebsiteAuditAgent:
    """Standalone LangGraph agent for website SEO auditing."""

    def __init__(self):
        self.graph = create_website_audit_agent()
        logger.info("Website Audit Agent initialized")

    async def process(
        self,
        url: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process a website SEO audit request.

        Args:
            url: URL to audit
            thread_id: Thread ID for conversation tracking (default: default)

        Returns:
            Dictionary containing the audit results and recommendations
        """
        try:
            # Validate inputs
            if not url:
                raise ValueError("URL must be provided")

            # Validate URL format
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("Invalid URL format")

            # Prepare initial state
            initial_state = {
                "messages": [],
                "url": url,
                "extracted_content": None,
                "html_structure": None,
                "seo_audit": None,
                "technical_audit": None,
                "content_audit": None,
                "recommendations": None,
                "overall_score": None,
                "status": "initialized",
                "error": None,
            }

            # Run the agent
            print("\n" + "=" * 80)
            print("🚀 Starting Website Audit Agent Workflow")
            print("=" * 80)
            config = {"configurable": {"thread_id": thread_id}}
            result = None

            async for event in self.graph.astream(initial_state, config):
                result = event
                # Log progress
                if "agent" in event:
                    status = event["agent"].get("status", "processing")
                    logger.info(f"Agent status: {status}")

            print("\n" + "=" * 80)
            print("✅ Workflow completed successfully!")
            print("=" * 80)

            # Get final state
            final_state = (
                result.get("agent", initial_state) if result else initial_state
            )

            # Check for errors
            if final_state.get("status") == "error":
                error_msg = final_state.get("error", "Unknown error occurred")
                raise ValueError(f"Agent processing failed: {error_msg}")

            # Return results
            return {
                "status": "success",
                "url": url,
                "extracted_content": final_state.get("extracted_content", {}),
                "seo_audit": final_state.get("seo_audit", {}),
                "recommendations": final_state.get("recommendations", {}),
                "overall_score": final_state.get("overall_score", 0),
                "processing_status": final_state.get("status", "unknown"),
            }

        except Exception as e:
            logger.error(f"Error processing website audit request: {str(e)}")
            raise


# ============================================================================
# Factory Function
# ============================================================================


def create_agent() -> WebsiteAuditAgent:
    """Factory function to create a new website audit agent instance."""
    return WebsiteAuditAgent()


# ============================================================================
# Utility Functions
# ============================================================================


def sanitize_filename(url: str, max_length: int = 100) -> str:
    """
    Sanitize a URL to be used as a filename.

    Args:
        url: The URL to sanitize
        max_length: Maximum length of the filename (default: 100)

    Returns:
        Sanitized filename-safe string
    """
    # Extract domain from URL
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")

    # Remove or replace invalid filename characters
    invalid_chars = '<>:"/\\|?*.'
    sanitized = domain.strip()

    for char in invalid_chars:
        sanitized = sanitized.replace(char, "-")

    # Replace multiple hyphens with single hyphen
    sanitized = re.sub(r"[\-]+", "-", sanitized)

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("-")

    return sanitized if sanitized else "website_audit"


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example of how to use the agent."""
    import os
    from datetime import datetime

    agent = create_agent()

    # Change this URL to audit a different website
    url = "https://www.9jacodekids.com/10-free-coding-apps-and-websites-for-kids/"

    result = await agent.process(url=url)

    # Create output folder
    try:
        # Try to get script directory if running as script
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir / "output"
    except NameError:
        # If __file__ not available (running as module), use current working directory
        output_dir = Path.cwd() / "output"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get data
    extracted_content = result.get("extracted_content", {})
    seo_audit = result.get("seo_audit", {})
    recommendations = result.get("recommendations", {})
    overall_score = result.get("overall_score", 0)

    # Create filename from URL
    sanitized_url = sanitize_filename(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save audit report as markdown
    report_file = output_dir / f"audit_{sanitized_url}_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Website SEO Audit Report\n\n")
        f.write(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**URL:** {url}\n\n")
        f.write(f"**Overall SEO Score:** {overall_score}/100\n\n")
        f.write("---\n\n")

        if extracted_content:
            f.write("## Page Information\n\n")
            f.write(f"- **Title:** {extracted_content.get('title', 'N/A')}\n")
            f.write(
                f"- **Title Length:** {extracted_content.get('title_length', 0)} characters\n"
            )
            f.write(
                f"- **Meta Description:** {extracted_content.get('meta_description', 'N/A')}\n"
            )
            f.write(
                f"- **Meta Description Length:** {extracted_content.get('meta_description_length', 0)} characters\n"
            )
            f.write(f"- **Word Count:** {extracted_content.get('word_count', 0)}\n")
            f.write(f"- **H1 Count:** {extracted_content.get('h1_count', 0)}\n")
            if extracted_content.get("headings", {}).get("h1"):
                f.write(
                    f"- **H1 Tags:** {', '.join(extracted_content['headings']['h1'])}\n"
                )
            f.write(f"\n### Images\n\n")
            f.write(
                f"- **Total:** {extracted_content.get('images', {}).get('total', 0)}\n"
            )
            f.write(
                f"- **With alt text:** {extracted_content.get('images', {}).get('with_alt', 0)}\n"
            )
            f.write(
                f"- **Without alt text:** {extracted_content.get('images', {}).get('without_alt', 0)}\n"
            )
            f.write(f"\n### Links\n\n")
            f.write(
                f"- **Total:** {extracted_content.get('links', {}).get('total', 0)}\n"
            )
            f.write(
                f"- **Internal:** {extracted_content.get('links', {}).get('internal', 0)}\n"
            )
            f.write(
                f"- **External:** {extracted_content.get('links', {}).get('external', 0)}\n\n"
            )
            f.write("---\n\n")

        if seo_audit:
            f.write("## SEO Audit Results\n\n")

            # Title Analysis
            if seo_audit.get("title_analysis"):
                f.write("### Title Analysis\n\n")
                title_analysis = seo_audit["title_analysis"]
                f.write(f"**Score:** {title_analysis.get('score', 0)}/100\n\n")
                if title_analysis.get("issues"):
                    f.write("#### Issues\n\n")
                    for issue in title_analysis["issues"]:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
                if title_analysis.get("suggestions"):
                    f.write("#### Suggestions\n\n")
                    for suggestion in title_analysis["suggestions"]:
                        f.write(f"- 💡 {suggestion}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Meta Description Analysis
            if seo_audit.get("meta_description_analysis"):
                f.write("### Meta Description Analysis\n\n")
                meta_analysis = seo_audit["meta_description_analysis"]
                f.write(f"**Score:** {meta_analysis.get('score', 0)}/100\n\n")
                if meta_analysis.get("issues"):
                    f.write("#### Issues\n\n")
                    for issue in meta_analysis["issues"]:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
                if meta_analysis.get("suggestions"):
                    f.write("#### Suggestions\n\n")
                    for suggestion in meta_analysis["suggestions"]:
                        f.write(f"- 💡 {suggestion}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Heading Structure
            if seo_audit.get("heading_structure"):
                f.write("### Heading Structure\n\n")
                heading = seo_audit["heading_structure"]
                f.write(f"**Score:** {heading.get('score', 0)}/100\n\n")
                if heading.get("issues"):
                    f.write("#### Issues\n\n")
                    for issue in heading["issues"]:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
                if heading.get("suggestions"):
                    f.write("#### Suggestions\n\n")
                    for suggestion in heading["suggestions"]:
                        f.write(f"- 💡 {suggestion}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Content Analysis
            if seo_audit.get("content_analysis"):
                f.write("### Content Analysis\n\n")
                content = seo_audit["content_analysis"]
                f.write(f"**Score:** {content.get('score', 0)}/100\n\n")
                if content.get("issues"):
                    f.write("#### Issues\n\n")
                    for issue in content["issues"]:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
                if content.get("suggestions"):
                    f.write("#### Suggestions\n\n")
                    for suggestion in content["suggestions"]:
                        f.write(f"- 💡 {suggestion}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Image Optimization
            if seo_audit.get("image_optimization"):
                f.write("### Image Optimization\n\n")
                images = seo_audit["image_optimization"]
                f.write(f"**Score:** {images.get('score', 0)}/100\n\n")
                if images.get("issues"):
                    f.write("#### Issues\n\n")
                    for issue in images["issues"]:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
                if images.get("suggestions"):
                    f.write("#### Suggestions\n\n")
                    for suggestion in images["suggestions"]:
                        f.write(f"- 💡 {suggestion}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Technical SEO
            if seo_audit.get("technical_seo"):
                f.write("### Technical SEO\n\n")
                technical = seo_audit["technical_seo"]
                f.write(f"**Score:** {technical.get('score', 0)}/100\n\n")
                if technical.get("issues"):
                    f.write("#### Issues\n\n")
                    for issue in technical["issues"]:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
                if technical.get("suggestions"):
                    f.write("#### Suggestions\n\n")
                    for suggestion in technical["suggestions"]:
                        f.write(f"- 💡 {suggestion}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Priority Fixes
            if seo_audit.get("priority_fixes"):
                f.write("### Priority Fixes\n\n")
                for i, fix in enumerate(seo_audit["priority_fixes"], 1):
                    f.write(f"#### {i}. {fix.get('issue', 'N/A')}\n\n")
                    f.write(
                        f"- **Priority:** {fix.get('priority', 'medium').upper()}\n"
                    )
                    f.write(f"- **Impact:** {fix.get('impact', 'N/A')}\n")
                    f.write(f"- **Solution:** {fix.get('solution', 'N/A')}\n\n")
                f.write("---\n\n")

            # Strengths
            if seo_audit.get("strengths"):
                f.write("### Strengths\n\n")
                for strength in seo_audit["strengths"]:
                    f.write(f"- ✅ {strength}\n")
                f.write("\n")
                f.write("---\n\n")

            # Quick Wins
            if seo_audit.get("quick_wins"):
                f.write("### Quick Wins\n\n")
                for win in seo_audit["quick_wins"]:
                    f.write(f"- ⚡ {win}\n")
                f.write("\n")
                f.write("---\n\n")

        if recommendations:
            f.write("## Action Plan\n\n")

            # Immediate Actions
            if recommendations.get("immediate_actions"):
                f.write("### Immediate Actions (Do First)\n\n")
                for i, action in enumerate(recommendations["immediate_actions"], 1):
                    f.write(f"#### {i}. {action.get('action', 'N/A')}\n\n")
                    f.write(
                        f"- **Expected Impact:** {action.get('expected_impact', 'N/A').upper()}\n"
                    )
                    f.write(
                        f"- **Implementation Time:** {action.get('implementation_time', 'N/A')}\n"
                    )
                    if action.get("steps"):
                        f.write(f"\n**Steps:**\n\n")
                        for step in action["steps"]:
                            f.write(f"1. {step}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Short-term Improvements
            if recommendations.get("short_term_improvements"):
                f.write("### Short-term Improvements (1-4 Weeks)\n\n")
                for i, action in enumerate(
                    recommendations["short_term_improvements"], 1
                ):
                    f.write(f"#### {i}. {action.get('action', 'N/A')}\n\n")
                    f.write(
                        f"- **Expected Impact:** {action.get('expected_impact', 'N/A').upper()}\n"
                    )
                    f.write(
                        f"- **Implementation Time:** {action.get('implementation_time', 'N/A')}\n"
                    )
                    if action.get("steps"):
                        f.write(f"\n**Steps:**\n\n")
                        for step in action["steps"]:
                            f.write(f"1. {step}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Long-term Strategy
            if recommendations.get("long_term_strategy"):
                f.write("### Long-term Strategy (1-6 Months)\n\n")
                for i, action in enumerate(recommendations["long_term_strategy"], 1):
                    f.write(f"#### {i}. {action.get('action', 'N/A')}\n\n")
                    f.write(
                        f"- **Expected Impact:** {action.get('expected_impact', 'N/A').upper()}\n"
                    )
                    f.write(
                        f"- **Implementation Time:** {action.get('implementation_time', 'N/A')}\n"
                    )
                    if action.get("steps"):
                        f.write(f"\n**Steps:**\n\n")
                        for step in action["steps"]:
                            f.write(f"1. {step}\n")
                    f.write("\n")
                f.write("---\n\n")

            # Summary
            f.write("### Summary\n\n")
            f.write(
                f"- **Estimated Overall Improvement:** {recommendations.get('estimated_overall_improvement', 'N/A')}\n"
            )
            f.write(f"- **Timeline:** {recommendations.get('timeline', 'N/A')}\n\n")

    print(f"\n✅ Audit report saved to: {report_file}")
    print(f"   Full path: {report_file.absolute()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
