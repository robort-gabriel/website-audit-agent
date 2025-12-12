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
    - audit_[domain]_[timestamp].pdf: Detailed SEO audit report in PDF format with scores and recommendations
"""

import logging
import json
import re
import time
import os
import sys
import asyncio
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

# Import PDFGenerator from root folder or current directory
PDFGenerator = None
try:
    # First try root folder (parent of website audit folder)
    script_dir = Path(__file__).parent.absolute()
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from pdf_generator import PDFGenerator

    logger.info("PDFGenerator imported successfully from root folder")
except ImportError:
    # Fallback: try current directory
    try:
        from pdf_generator import PDFGenerator

        logger.info("PDFGenerator imported successfully from current directory")
    except ImportError:
        PDFGenerator = None
        logger.warning(
            "PDFGenerator not found. PDF generation will be disabled. "
            "Make sure pdf_generator.py is in the root folder or current directory."
        )
except Exception as e:
    PDFGenerator = None
    logger.warning(
        f"Error importing PDFGenerator. PDF generation will be disabled. Error: {str(e)}"
    )

# Get OpenAI API key from environment (will be checked when LLMService is initialized)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get OpenAI model from environment (defaults to gpt-4o-2024-11-20)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20")


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
    speed_audit: Optional[Dict[str, Any]]
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


@tool
async def check_page_speed(url: str) -> str:
    """
    Check page load speed and performance metrics for a webpage.

    Args:
        url: The URL to check speed for

    Returns:
        JSON string containing speed metrics including load time, response time, and performance analysis
    """
    try:
        logger.info(f"Checking page speed for URL: {url}")

        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return json.dumps({"error": "Invalid URL format"})

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        }

        # Measure response time (including content download)
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                # Time to first byte (TTFB)
                ttfb = time.time() - start_time

                # Get response size and measure total load time
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        response_size = int(content_length)
                        # Still read content to measure actual download time
                        content = await response.read()
                        total_time = time.time() - start_time
                        content = None  # Free memory
                    except:
                        response_size = None
                        content = await response.read()
                        response_size = len(content)
                        total_time = time.time() - start_time
                        content = None  # Free memory
                else:
                    # Read content to measure size and total time
                    content = await response.read()
                    response_size = len(content)
                    total_time = time.time() - start_time
                    content = None  # Free memory

                # Use total_time for scoring (includes full page download)
                response_time = total_time

                status_code = response.status

                # Check for compression
                content_encoding = response.headers.get("Content-Encoding", "")
                is_compressed = content_encoding.lower() in ["gzip", "br", "deflate"]

                # Check cache headers
                cache_control = response.headers.get("Cache-Control", "")
                expires = response.headers.get("Expires", "")
                has_cache = bool(cache_control or expires)

                # Check for CDN
                server = response.headers.get("Server", "")
                cdn_headers = ["cloudflare", "cloudfront", "fastly", "akamai", "maxcdn"]
                uses_cdn = any(cdn in server.lower() for cdn in cdn_headers)

                # Check HTTP version
                http_version = f"HTTP/{response.version.major}.{response.version.minor}"

                # Calculate performance score (0-100)
                # Response time scoring with more granular thresholds
                # <0.5s = 100, <1s = 95, <1.5s = 85, <2s = 75, <3s = 60, <5s = 40, <10s = 20, >=10s = 10
                if response_time < 0.5:
                    speed_score = 100
                elif response_time < 1.0:
                    speed_score = 95
                elif response_time < 1.5:
                    speed_score = 85
                elif response_time < 2.0:
                    speed_score = 75
                elif response_time < 3.0:
                    speed_score = 60
                elif response_time < 5.0:
                    speed_score = 40
                elif response_time < 10.0:
                    speed_score = 20
                else:
                    speed_score = 10

                # Adjust score based on optimizations (smaller bonuses to avoid inflating scores)
                optimization_bonus = 0
                if is_compressed:
                    optimization_bonus += 3
                if has_cache:
                    optimization_bonus += 2
                if uses_cdn:
                    optimization_bonus += 2

                speed_score = min(100, speed_score + optimization_bonus)

                # Response size scoring (smaller is better)
                # More granular size scoring
                if response_size:
                    size_mb = response_size / (1024 * 1024)
                    if size_mb < 0.5:
                        size_score = 100
                    elif size_mb < 1.0:
                        size_score = 95
                    elif size_mb < 2.0:
                        size_score = 80
                    elif size_mb < 3.0:
                        size_score = 60
                    elif size_mb < 5.0:
                        size_score = 40
                    elif size_mb < 10.0:
                        size_score = 20
                    else:
                        size_score = 10
                else:
                    size_score = 50  # Unknown size

                # Overall performance score (weighted average: 75% speed, 25% size)
                overall_speed_score = int((speed_score * 0.75) + (size_score * 0.25))

                result = {
                    "url": url,
                    "response_time_seconds": round(response_time, 3),
                    "response_time_ms": round(response_time * 1000, 2),
                    "ttfb_seconds": round(ttfb, 3),
                    "ttfb_ms": round(ttfb * 1000, 2),
                    "status_code": status_code,
                    "response_size_bytes": response_size,
                    "response_size_mb": (
                        round(response_size / (1024 * 1024), 2)
                        if response_size
                        else None
                    ),
                    "is_compressed": is_compressed,
                    "compression_type": content_encoding if is_compressed else "none",
                    "has_cache_headers": has_cache,
                    "cache_control": cache_control if cache_control else None,
                    "uses_cdn": uses_cdn,
                    "server": server if server else "unknown",
                    "http_version": http_version,
                    "speed_score": speed_score,
                    "size_score": size_score,
                    "overall_speed_score": overall_speed_score,
                    "performance_rating": (
                        "Excellent"
                        if overall_speed_score >= 90
                        else (
                            "Good"
                            if overall_speed_score >= 70
                            else (
                                "Fair"
                                if overall_speed_score >= 50
                                else (
                                    "Poor" if overall_speed_score >= 30 else "Very Poor"
                                )
                            )
                        )
                    ),
                }

                return json.dumps(result, indent=2)

    except asyncio.TimeoutError:
        logger.error(f"Timeout checking page speed for URL: {url}")
        return json.dumps(
            {
                "error": "Request timeout",
                "url": url,
                "response_time_seconds": None,
                "overall_speed_score": 0,
                "performance_rating": "Timeout",
            }
        )
    except Exception as e:
        logger.error(f"Error checking page speed: {str(e)}")
        return json.dumps({"error": f"Failed to check page speed: {str(e)}"})


# ============================================================================
# LLM Service Functions
# ============================================================================


class LLMService:
    """Service for interacting with the LLM."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is not set. Please set OPENAI_API_KEY environment variable or pass it as a parameter."
            )

        # Use provided model, environment variable, or default
        self.model = model or OPENAI_MODEL
        logger.info(f"Using OpenAI model: {self.model}")

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=1,
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


async def check_speed_node(state: WebsiteAuditState) -> WebsiteAuditState:
    """Check page load speed and performance metrics."""
    try:
        print("\n" + "=" * 80)
        print("⚡ NODE: check_speed_node - Checking page speed")
        print("=" * 80)
        logger.info("Checking page speed")

        url = state.get("url", "")
        if not url:
            return {
                **state,
                "status": "error",
                "error": "No URL provided for speed check",
            }

        speed_json = await check_page_speed.ainvoke(url)
        speed_data = json.loads(speed_json)

        if "error" in speed_data:
            print(
                f"⚠️  Warning: Speed check encountered an issue: {speed_data['error']}"
            )
            # Don't fail the workflow, just log the warning
            speed_data["overall_speed_score"] = 0
            speed_data["performance_rating"] = "Error"

        response_time = speed_data.get("response_time_seconds", 0)
        speed_score = speed_data.get("overall_speed_score", 0)
        print(
            f"✅ Speed check completed (Response time: {response_time:.3f}s, Score: {speed_score}/100)"
        )

        return {
            **state,
            "speed_audit": speed_data,
            "status": "speed_checked",
        }
    except Exception as e:
        print(f"❌ Error in check_speed_node: {str(e)}")
        logger.error(f"Error in check_speed_node: {str(e)}")
        # Don't fail the workflow, just log the error
        return {
            **state,
            "speed_audit": {"error": str(e), "overall_speed_score": 0},
            "status": "speed_checked",
        }


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
            # Move to SEO analysis (can run in parallel with speed check, but we'll do SEO first)
            return await analyze_seo_node(state)

        elif current_status == "seo_analyzed":
            # Check page speed after SEO analysis
            return await check_speed_node(state)

        elif current_status == "speed_checked":
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

    # Compile the graph (no memory/checkpointer needed for one-time analysis)
    graph = workflow.compile()

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
    ) -> Dict[str, Any]:
        """
        Process a website SEO audit request.

        Args:
            url: URL to audit

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
                "speed_audit": None,
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
            result = None

            async for event in self.graph.astream(initial_state):
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
                "speed_audit": final_state.get("speed_audit", {}),
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
# Report Generation Functions
# ============================================================================


def generate_audit_report_markdown(
    url: str,
    extracted_content: Dict[str, Any],
    seo_audit: Dict[str, Any],
    recommendations: Dict[str, Any],
    overall_score: int,
    speed_audit: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate markdown content for the audit report."""
    markdown_content = []

    markdown_content.append(f"**URL:** {url}\n\n")
    markdown_content.append(f"**Overall SEO Score:** {overall_score}/100\n\n")

    # Add speed score if available
    if speed_audit and not speed_audit.get("error"):
        speed_score = speed_audit.get("overall_speed_score", 0)
        markdown_content.append(f"**Page Speed Score:** {speed_score}/100\n\n")

    if extracted_content:
        markdown_content.append("## Page Information\n\n")
        markdown_content.append(
            f"- **Title:** {extracted_content.get('title', 'N/A')}\n"
        )
        markdown_content.append(
            f"- **Title Length:** {extracted_content.get('title_length', 0)} characters\n"
        )
        markdown_content.append(
            f"- **Meta Description:** {extracted_content.get('meta_description', 'N/A')}\n"
        )
        markdown_content.append(
            f"- **Meta Description Length:** {extracted_content.get('meta_description_length', 0)} characters\n"
        )
        markdown_content.append(
            f"- **Word Count:** {extracted_content.get('word_count', 0)}\n"
        )
        markdown_content.append(
            f"- **H1 Count:** {extracted_content.get('h1_count', 0)}\n"
        )
        if extracted_content.get("headings", {}).get("h1"):
            markdown_content.append(
                f"- **H1 Tags:** {', '.join(extracted_content['headings']['h1'])}\n"
            )
        markdown_content.append(f"\n### Images\n\n")
        markdown_content.append(
            f"- **Total:** {extracted_content.get('images', {}).get('total', 0)}\n"
        )
        markdown_content.append(
            f"- **With alt text:** {extracted_content.get('images', {}).get('with_alt', 0)}\n"
        )
        markdown_content.append(
            f"- **Without alt text:** {extracted_content.get('images', {}).get('without_alt', 0)}\n"
        )
        markdown_content.append(f"\n### Links\n\n")
        markdown_content.append(
            f"- **Total:** {extracted_content.get('links', {}).get('total', 0)}\n"
        )
        markdown_content.append(
            f"- **Internal:** {extracted_content.get('links', {}).get('internal', 0)}\n"
        )
        markdown_content.append(
            f"- **External:** {extracted_content.get('links', {}).get('external', 0)}\n\n"
        )

    if seo_audit:
        markdown_content.append("## SEO Audit Results\n\n")

        # Title Analysis
        if seo_audit.get("title_analysis"):
            markdown_content.append("### Title Analysis\n\n")
            title_analysis = seo_audit["title_analysis"]
            markdown_content.append(
                f"**Score:** {title_analysis.get('score', 0)}/100\n\n"
            )
            if title_analysis.get("issues"):
                markdown_content.append("#### Issues\n\n")
                for issue in title_analysis["issues"]:
                    markdown_content.append(
                        f'- <span class="issue-item">{issue}</span>\n'
                    )
                markdown_content.append("\n")
            if title_analysis.get("suggestions"):
                markdown_content.append("#### Suggestions\n\n")
                for suggestion in title_analysis["suggestions"]:
                    markdown_content.append(
                        f'- <span class="suggestion-item">{suggestion}</span>\n'
                    )
                markdown_content.append("\n")

        # Meta Description Analysis
        if seo_audit.get("meta_description_analysis"):
            markdown_content.append("### Meta Description Analysis\n\n")
            meta_analysis = seo_audit["meta_description_analysis"]
            markdown_content.append(
                f"**Score:** {meta_analysis.get('score', 0)}/100\n\n"
            )
            if meta_analysis.get("issues"):
                markdown_content.append("#### Issues\n\n")
                for issue in meta_analysis["issues"]:
                    markdown_content.append(
                        f'- <span class="issue-item">{issue}</span>\n'
                    )
                markdown_content.append("\n")
            if meta_analysis.get("suggestions"):
                markdown_content.append("#### Suggestions\n\n")
                for suggestion in meta_analysis["suggestions"]:
                    markdown_content.append(
                        f'- <span class="suggestion-item">{suggestion}</span>\n'
                    )
                markdown_content.append("\n")

        # Heading Structure
        if seo_audit.get("heading_structure"):
            markdown_content.append("### Heading Structure\n\n")
            heading = seo_audit["heading_structure"]
            markdown_content.append(f"**Score:** {heading.get('score', 0)}/100\n\n")
            if heading.get("issues"):
                markdown_content.append("#### Issues\n\n")
                for issue in heading["issues"]:
                    markdown_content.append(
                        f'- <span class="issue-item">{issue}</span>\n'
                    )
                markdown_content.append("\n")
            if heading.get("suggestions"):
                markdown_content.append("#### Suggestions\n\n")
                for suggestion in heading["suggestions"]:
                    markdown_content.append(
                        f'- <span class="suggestion-item">{suggestion}</span>\n'
                    )
                markdown_content.append("\n")

        # Content Analysis
        if seo_audit.get("content_analysis"):
            markdown_content.append("### Content Analysis\n\n")
            content = seo_audit["content_analysis"]
            markdown_content.append(f"**Score:** {content.get('score', 0)}/100\n\n")
            if content.get("issues"):
                markdown_content.append("#### Issues\n\n")
                for issue in content["issues"]:
                    markdown_content.append(
                        f'- <span class="issue-item">{issue}</span>\n'
                    )
                markdown_content.append("\n")
            if content.get("suggestions"):
                markdown_content.append("#### Suggestions\n\n")
                for suggestion in content["suggestions"]:
                    markdown_content.append(
                        f'- <span class="suggestion-item">{suggestion}</span>\n'
                    )
                markdown_content.append("\n")

        # Image Optimization
        if seo_audit.get("image_optimization"):
            markdown_content.append("### Image Optimization\n\n")
            images = seo_audit["image_optimization"]
            markdown_content.append(f"**Score:** {images.get('score', 0)}/100\n\n")
            if images.get("issues"):
                markdown_content.append("#### Issues\n\n")
                for issue in images["issues"]:
                    markdown_content.append(
                        f'- <span class="issue-item">{issue}</span>\n'
                    )
                markdown_content.append("\n")
            if images.get("suggestions"):
                markdown_content.append("#### Suggestions\n\n")
                for suggestion in images["suggestions"]:
                    markdown_content.append(
                        f'- <span class="suggestion-item">{suggestion}</span>\n'
                    )
                markdown_content.append("\n")

        # Technical SEO
        if seo_audit.get("technical_seo"):
            markdown_content.append("### Technical SEO\n\n")
            technical = seo_audit["technical_seo"]
            markdown_content.append(f"**Score:** {technical.get('score', 0)}/100\n\n")
            if technical.get("issues"):
                markdown_content.append("#### Issues\n\n")
                for issue in technical["issues"]:
                    markdown_content.append(
                        f'- <span class="issue-item">{issue}</span>\n'
                    )
                markdown_content.append("\n")
            if technical.get("suggestions"):
                markdown_content.append("#### Suggestions\n\n")
                for suggestion in technical["suggestions"]:
                    markdown_content.append(
                        f'- <span class="suggestion-item">{suggestion}</span>\n'
                    )
                markdown_content.append("\n")

    # Speed Audit Section
    if speed_audit and not speed_audit.get("error"):
        markdown_content.append("## Page Speed Analysis\n\n")
        markdown_content.append(
            f"**Overall Speed Score:** {speed_audit.get('overall_speed_score', 0)}/100\n\n"
        )
        markdown_content.append(
            f"**Performance Rating:** {speed_audit.get('performance_rating', 'N/A')}\n\n"
        )

        markdown_content.append("### Speed Metrics\n\n")
        markdown_content.append(
            f"- **Response Time:** {speed_audit.get('response_time_seconds', 0):.3f} seconds ({speed_audit.get('response_time_ms', 0):.2f} ms)\n"
        )
        if speed_audit.get("response_size_bytes"):
            markdown_content.append(
                f"- **Page Size:** {speed_audit.get('response_size_mb', 0):.2f} MB ({speed_audit.get('response_size_bytes', 0):,} bytes)\n"
            )
        markdown_content.append(
            f"- **HTTP Version:** {speed_audit.get('http_version', 'N/A')}\n"
        )
        markdown_content.append(
            f"- **Status Code:** {speed_audit.get('status_code', 'N/A')}\n\n"
        )

        markdown_content.append("### Performance Optimizations\n\n")
        markdown_content.append(
            f"- **Compression:** {'✅ Enabled' if speed_audit.get('is_compressed') else '❌ Not enabled'}"
        )
        if speed_audit.get("is_compressed"):
            markdown_content.append(
                f" ({speed_audit.get('compression_type', 'unknown')})"
            )
        markdown_content.append("\n")
        markdown_content.append(
            f"- **Cache Headers:** {'✅ Present' if speed_audit.get('has_cache_headers') else '❌ Missing'}\n"
        )
        markdown_content.append(
            f"- **CDN:** {'✅ Detected' if speed_audit.get('uses_cdn') else '❌ Not detected'}"
        )
        if speed_audit.get("uses_cdn") and speed_audit.get("server"):
            markdown_content.append(f" ({speed_audit.get('server', '')})")
        markdown_content.append("\n\n")

        # Speed recommendations
        speed_issues = []
        speed_suggestions = []

        if speed_audit.get("response_time_seconds", 0) > 3.0:
            speed_issues.append(
                f"Slow response time ({speed_audit.get('response_time_seconds', 0):.3f}s) - should be under 2 seconds"
            )
            speed_suggestions.append(
                "Optimize server response time, use CDN, enable caching"
            )

        if not speed_audit.get("is_compressed"):
            speed_issues.append("Content compression not enabled")
            speed_suggestions.append(
                "Enable GZIP or Brotli compression to reduce page size"
            )

        if not speed_audit.get("has_cache_headers"):
            speed_issues.append("Cache headers missing")
            speed_suggestions.append(
                "Add Cache-Control and Expires headers for static resources"
            )

        if (
            speed_audit.get("response_size_mb", 0)
            and speed_audit.get("response_size_mb", 0) > 3.0
        ):
            speed_issues.append(
                f"Large page size ({speed_audit.get('response_size_mb', 0):.2f} MB) - should be under 2 MB"
            )
            speed_suggestions.append(
                "Optimize images, minify CSS/JS, remove unused resources"
            )

        if speed_issues:
            markdown_content.append("#### Speed Issues\n\n")
            for issue in speed_issues:
                markdown_content.append(f'- <span class="issue-item">{issue}</span>\n')
            markdown_content.append("\n")

        if speed_suggestions:
            markdown_content.append("#### Speed Optimization Suggestions\n\n")
            for suggestion in speed_suggestions:
                markdown_content.append(
                    f'- <span class="suggestion-item">{suggestion}</span>\n'
                )
            markdown_content.append("\n")

    if seo_audit:
        # Priority Fixes
        if seo_audit.get("priority_fixes"):
            markdown_content.append("### Priority Fixes\n\n")
            for i, fix in enumerate(seo_audit["priority_fixes"], 1):
                markdown_content.append(f"#### {i}. {fix.get('issue', 'N/A')}\n\n")
                markdown_content.append(
                    f"- **Priority:** {fix.get('priority', 'medium').upper()}\n"
                )
                markdown_content.append(f"- **Impact:** {fix.get('impact', 'N/A')}\n")
                markdown_content.append(
                    f"- **Solution:** {fix.get('solution', 'N/A')}\n\n"
                )

        # Strengths
        if seo_audit.get("strengths"):
            markdown_content.append("### Strengths\n\n")
            for strength in seo_audit["strengths"]:
                markdown_content.append(
                    f'- <span class="strength-item">{strength}</span>\n'
                )
            markdown_content.append("\n")

        # Quick Wins
        if seo_audit.get("quick_wins"):
            markdown_content.append("### Quick Wins\n\n")
            for win in seo_audit["quick_wins"]:
                markdown_content.append(
                    f'- <span class="quick-win-item">{win}</span>\n'
                )
            markdown_content.append("\n")

    if recommendations:
        markdown_content.append("## Action Plan\n\n")

        # Immediate Actions
        if recommendations.get("immediate_actions"):
            markdown_content.append("### Immediate Actions (Do First)\n\n")
            for i, action in enumerate(recommendations["immediate_actions"], 1):
                markdown_content.append(f"#### {i}. {action.get('action', 'N/A')}\n\n")
                markdown_content.append(
                    f"- **Expected Impact:** {action.get('expected_impact', 'N/A').upper()}\n"
                )
                markdown_content.append(
                    f"- **Implementation Time:** {action.get('implementation_time', 'N/A')}\n"
                )
                if action.get("steps"):
                    markdown_content.append(f"\n**Steps:**\n\n")
                    for step in action["steps"]:
                        markdown_content.append(f"1. {step}\n")
                markdown_content.append("\n")

        # Short-term Improvements
        if recommendations.get("short_term_improvements"):
            markdown_content.append("### Short-term Improvements (1-4 Weeks)\n\n")
            for i, action in enumerate(recommendations["short_term_improvements"], 1):
                markdown_content.append(f"#### {i}. {action.get('action', 'N/A')}\n\n")
                markdown_content.append(
                    f"- **Expected Impact:** {action.get('expected_impact', 'N/A').upper()}\n"
                )
                markdown_content.append(
                    f"- **Implementation Time:** {action.get('implementation_time', 'N/A')}\n"
                )
                if action.get("steps"):
                    markdown_content.append(f"\n**Steps:**\n\n")
                    for step in action["steps"]:
                        markdown_content.append(f"1. {step}\n")
                markdown_content.append("\n")

        # Long-term Strategy
        if recommendations.get("long_term_strategy"):
            markdown_content.append("### Long-term Strategy (1-6 Months)\n\n")
            for i, action in enumerate(recommendations["long_term_strategy"], 1):
                markdown_content.append(f"#### {i}. {action.get('action', 'N/A')}\n\n")
                markdown_content.append(
                    f"- **Expected Impact:** {action.get('expected_impact', 'N/A').upper()}\n"
                )
                markdown_content.append(
                    f"- **Implementation Time:** {action.get('implementation_time', 'N/A')}\n"
                )
                if action.get("steps"):
                    markdown_content.append(f"\n**Steps:**\n\n")
                    for step in action["steps"]:
                        markdown_content.append(f"1. {step}\n")
                markdown_content.append("\n")

        # Summary
        markdown_content.append("### Summary\n\n")
        markdown_content.append(
            f"- **Estimated Overall Improvement:** {recommendations.get('estimated_overall_improvement', 'N/A')}\n"
        )
        markdown_content.append(
            f"- **Timeline:** {recommendations.get('timeline', 'N/A')}\n\n"
        )

    return "".join(markdown_content)


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
    speed_audit = result.get("speed_audit", {})
    recommendations = result.get("recommendations", {})
    overall_score = result.get("overall_score", 0)

    # Create filename from URL
    sanitized_url = sanitize_filename(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate markdown content for PDF
    report_markdown = generate_audit_report_markdown(
        url=url,
        extracted_content=extracted_content,
        seo_audit=seo_audit,
        recommendations=recommendations,
        overall_score=overall_score,
        speed_audit=speed_audit,
    )

    # Generate PDF using PDFGenerator
    if PDFGenerator is None:
        # Fallback to markdown if PDFGenerator is not available
        report_file = output_dir / f"audit_{sanitized_url}_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Website SEO Audit Report\n\n")
            f.write(
                f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(report_markdown)
        print(f"\n✅ Audit report saved to: {report_file}")
        print(f"   Full path: {report_file.absolute()}")
        print("   Note: PDF generation is not available. Saved as markdown instead.")
    else:
        try:
            # Prepare article data for PDFGenerator
            page_title = (
                extracted_content.get("title", "Website SEO Audit Report")
                if extracted_content
                else "Website SEO Audit Report"
            )
            meta_description = (
                extracted_content.get("meta_description", f"SEO audit report for {url}")
                if extracted_content
                else f"SEO audit report for {url}"
            )

            article_data = {
                "title": f"Website SEO Audit Report - {page_title}",
                "description": meta_description,
                "content": report_markdown,
                "keywords": [],
                "meta_info": {"thumbnail_url": "", "faqs": []},
                "created_at": datetime.now().isoformat(),
                "word_count": (
                    extracted_content.get("word_count", 0) if extracted_content else 0
                ),
                "readability_level": "General",
                "target_audience": "General",
                "article_tone": "Professional",
            }

            # Generate PDF
            pdf_generator = PDFGenerator()
            pdf_bytes = pdf_generator.generate_article_pdf(article_data)

            # Save PDF file
            report_file = output_dir / f"audit_{sanitized_url}_{timestamp}.pdf"
            with open(report_file, "wb") as f:
                f.write(pdf_bytes)

            print(f"\n✅ Audit report saved to: {report_file}")
            print(f"   Full path: {report_file.absolute()}")
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            # Fallback to markdown on error
            report_file = output_dir / f"audit_{sanitized_url}_{timestamp}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# Website SEO Audit Report\n\n")
                f.write(
                    f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                f.write(report_markdown)
            print(
                f"\n⚠️  PDF generation failed. Audit report saved as markdown: {report_file}"
            )
            print(f"   Error: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
