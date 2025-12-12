"""
FastAPI Application for Website SEO Audit Agent

Production-ready FastAPI application with security features:
- API key authentication
- Rate limiting
- Input validation
- CORS configuration
- Comprehensive error handling
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from website_audit_agent import create_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rate limiter configuration
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
limiter = Limiter(key_func=get_remote_address)

# API Key Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    logger.warning(
        "API_KEY not set in environment variables. API will be accessible without authentication."
    )


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verify API key from request header.

    Args:
        api_key: API key from request header

    Returns:
        True if API key is valid

    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not API_KEY:
        # If no API key is configured, allow all requests
        return True

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing. Please provide X-API-Key header.",
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return True


# Request/Response Models
class AuditRequest(BaseModel):
    """Request model for website SEO audit."""

    url: HttpUrl = Field(
        ...,
        description="Website URL to audit for SEO",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> str:
        """Convert HttpUrl to string."""
        return str(v)


class AuditResponse(BaseModel):
    """Response model for website audit results."""

    status: str
    url: str
    extracted_content: Dict[str, Any]
    seo_audit: Dict[str, Any]
    recommendations: Dict[str, Any]
    overall_score: int
    processing_status: str
    pdf_path: Optional[str] = None
    pdf_filename: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    error_type: Optional[str] = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app."""
    logger.info("Starting Website SEO Audit API...")
    yield
    logger.info("Shutting down Website SEO Audit API...")


# Create FastAPI app
app = FastAPI(
    title="Website SEO Audit API",
    description="Production-ready API for performing comprehensive SEO audits on websites",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check():
    """
    Health check endpoint to verify API is running.

    Returns:
        Health status and API version
    """
    return HealthResponse(
        status="healthy",
        message="Website SEO Audit API is running",
        version="1.0.0",
    )


@app.post(
    "/api/v1/audit",
    response_model=AuditResponse,
    status_code=status.HTTP_200_OK,
    tags=["SEO Audit"],
    summary="Perform SEO audit on a website",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE}/minute"
)  # Rate limit from environment variable
async def audit_website(
    request_data: AuditRequest,
    request: Request,
):
    """
    Perform a comprehensive SEO audit on a website.

    This endpoint:
    - Extracts content and SEO elements from the webpage
    - Analyzes SEO factors (title, meta description, headings, images, links, etc.)
    - Generates actionable recommendations
    - Creates a PDF report with the audit results
    - Returns comprehensive audit data

    Args:
        request_data: Audit request with URL
        request: FastAPI Request object for rate limiting

    Returns:
        AuditResponse with audit results and PDF path

    Raises:
        HTTPException: If audit fails or request is invalid
    """
    try:
        logger.info(f"Received SEO audit request: url='{request_data.url}'")

        # Create agent instance
        agent = create_agent()

        # Process the audit request
        result = await agent.process(
            url=request_data.url,
        )

        # Generate PDF report
        pdf_path = None
        pdf_filename = None
        try:
            from pathlib import Path
            from datetime import datetime
            from website_audit_agent import (
                generate_audit_report_markdown,
                sanitize_filename,
            )

            # Import PDFGenerator from root folder
            import sys

            script_dir = Path(__file__).parent.absolute()
            parent_dir = script_dir.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from pdf_generator import PDFGenerator

            if PDFGenerator is not None:
                # Create output folder
                output_dir = script_dir / "output"
                output_dir.mkdir(exist_ok=True)

                # Generate markdown content for PDF
                report_markdown = generate_audit_report_markdown(
                    url=request_data.url,
                    extracted_content=result.get("extracted_content", {}),
                    seo_audit=result.get("seo_audit", {}),
                    recommendations=result.get("recommendations", {}),
                    overall_score=result.get("overall_score", 0),
                    speed_audit=result.get("speed_audit", {}),
                )

                # Prepare article data for PDFGenerator
                extracted_content = result.get("extracted_content", {})
                page_title = (
                    extracted_content.get("title", "Website SEO Audit Report")
                    if extracted_content
                    else "Website SEO Audit Report"
                )
                meta_description = (
                    extracted_content.get(
                        "meta_description", f"SEO audit report for {request_data.url}"
                    )
                    if extracted_content
                    else f"SEO audit report for {request_data.url}"
                )

                article_data = {
                    "title": f"Website SEO Audit Report - {page_title}",
                    "description": meta_description,
                    "content": report_markdown,
                    "keywords": [],
                    "meta_info": {"thumbnail_url": "", "faqs": []},
                    "created_at": datetime.now().isoformat(),
                    "word_count": (
                        extracted_content.get("word_count", 0)
                        if extracted_content
                        else 0
                    ),
                    "readability_level": "General",
                    "target_audience": "General",
                    "article_tone": "Professional",
                }

                # Generate PDF
                pdf_generator = PDFGenerator()
                pdf_bytes = pdf_generator.generate_article_pdf(article_data)

                # Save PDF file
                sanitized_url = sanitize_filename(request_data.url)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"audit_{sanitized_url}_{timestamp}.pdf"
                pdf_path = output_dir / pdf_filename

                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                logger.info(f"PDF report generated: {pdf_path}")
        except Exception as pdf_error:
            logger.warning(f"PDF generation failed: {str(pdf_error)}")
            # Continue without PDF - not a critical error

        logger.info(
            f"Successfully completed SEO audit for URL: {request_data.url} "
            f"(Score: {result.get('overall_score', 0)}/100)"
        )

        return AuditResponse(
            status=result.get("status", "success"),
            url=result.get("url", request_data.url),
            extracted_content=result.get("extracted_content", {}),
            seo_audit=result.get("seo_audit", {}),
            speed_audit=result.get("speed_audit", {}),
            recommendations=result.get("recommendations", {}),
            overall_score=result.get("overall_score", 0),
            processing_status=result.get("processing_status", "unknown"),
            pdf_path=str(pdf_path) if pdf_path else None,
            pdf_filename=pdf_filename,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error performing SEO audit: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SEO audit failed: {str(e)}",
        )


@app.get(
    "/",
    tags=["Root"],
    summary="API root endpoint",
)
async def root():
    """
    Root endpoint with API information.

    Returns:
        API information and available endpoints
    """
    return {
        "name": "Website SEO Audit API",
        "version": "1.0.0",
        "description": "Production-ready API for performing comprehensive SEO audits on websites",
        "endpoints": {
            "health": "/health",
            "audit": "/api/v1/audit",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 9000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true",
        log_level="info",
    )
