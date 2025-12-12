import os
import base64
from datetime import datetime
from typing import Dict, Any, Optional
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import jinja2
from markdown import markdown
import bleach
import re


class PDFGenerator:
    def __init__(self):
        self.font_config = FontConfiguration()

    def generate_article_pdf(self, article: Dict[str, Any]) -> bytes:
        """Generate a professional PDF from article data"""

        # Convert markdown to HTML with proper styling and remove H1
        content_html = self._convert_markdown_to_html(article.get("content", ""))

        # Prepare template data
        template_data = {
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": content_html,
            "keywords": article.get("keywords", []),
            "thumbnail": article.get("meta_info", {}).get("thumbnail_url", ""),
            "created_at": self._format_date(article.get("created_at", "")),
            "word_count": article.get("word_count", 0),
            "readability_level": article.get("readability_level", "General"),
            "target_audience": article.get("target_audience", "General"),
            "article_tone": article.get("article_tone", "Neutral"),
            "faqs": article.get("meta_info", {}).get("faqs", []),
            "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        }

        # Generate HTML
        html_content = self._render_template(template_data)

        # Generate CSS
        css_content = self._get_pdf_styles()

        # Create PDF
        html_doc = HTML(string=html_content)
        css_doc = CSS(string=css_content, font_config=self.font_config)

        pdf_bytes = html_doc.write_pdf(
            stylesheets=[css_doc], font_config=self.font_config
        )

        return pdf_bytes

    def _format_date(self, date_str: str) -> str:
        """Format date string for display"""
        try:
            if isinstance(date_str, str):
                # Try different date formats
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%d %H:%M:%S",
                ]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime("%B %d, %Y")
                    except ValueError:
                        continue
            return str(date_str)
        except:
            return str(date_str)

    def _convert_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown to clean HTML and remove H1 tags"""
        if not markdown_content:
            return ""

        # Convert markdown to HTML
        html = markdown(
            markdown_content,
            extensions=[
                "markdown.extensions.tables",
                "markdown.extensions.fenced_code",
                "markdown.extensions.toc",
                "markdown.extensions.codehilite",
                "markdown.extensions.attr_list",
            ],
        )

        # Remove H1 tags from content since we use meta title
        html = re.sub(r"<h1[^>]*>.*?</h1>", "", html, flags=re.IGNORECASE | re.DOTALL)

        # Sanitize HTML
        allowed_tags = [
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "br",
            "strong",
            "em",
            "u",
            "s",
            "ul",
            "ol",
            "li",
            "blockquote",
            "code",
            "pre",
            "table",
            "thead",
            "tbody",
            "tr",
            "th",
            "td",
            "a",
            "img",
            "div",
            "span",
        ]

        allowed_attributes = {
            "a": ["href", "title"],
            "img": ["src", "alt", "title", "width", "height", "style"],
            "span": ["class"],
            "*": ["class", "id"],
        }

        clean_html = bleach.clean(
            html, tags=allowed_tags, attributes=allowed_attributes
        )
        return clean_html

    def _render_template(self, data: Dict[str, Any]) -> str:
        """Render HTML template with article data"""
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
        </head>
        <body>
            <!-- Header -->
            <header class="pdf-header">
                <div class="header-content">
                    {% if thumbnail %}
                    <div class="article-thumbnail">
                        <img src="{{ thumbnail }}" alt="{{ title }}" class="thumbnail-image" />
                    </div>
                    {% endif %}
                    <h1 class="article-title">{{ title }}</h1>
                    {% if description %}
                    <p class="article-description">{{ description }}</p>
                    {% endif %}
                </div>
            </header>

            <!-- Main Content -->
            <main class="pdf-content">
                {{ content | safe }}
            </main>

            <!-- Keywords Section -->
            {% if keywords %}
            <section class="keywords-section">
                <h2>Keywords</h2>
                <div class="keywords-list">
                    {% for keyword in keywords %}
                    <span class="keyword-tag">{{ keyword }}</span>
                    {% endfor %}
                </div>
            </section>
            {% endif %}

            <!-- FAQs Section -->
            {% if faqs %}
            <section class="faqs-section">
                <h2>Frequently Asked Questions</h2>
                {% for faq in faqs %}
                <div class="faq-item">
                    <h3 class="faq-question">Q: {{ faq.question }}</h3>
                    <p class="faq-answer">A: {{ faq.answer }}</p>
                </div>
                {% endfor %}
            </section>
            {% endif %}

        </body>
        </html>
        """

        template = jinja2.Template(template_str)
        return template.render(**data)

    def _get_pdf_styles(self) -> str:
        """Get comprehensive CSS styles for PDF"""
        return """
        @page {
            size: A4;
            margin: 0.75in;
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
                border-top: 1px solid #ddd;
                padding-top: 5pt;
            }
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            font-size: 11pt;
        }

        /* Header Styles */
        .pdf-header {
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid #2c5aa0;
        }

        .header-content {
            text-align: center;
        }

        .article-thumbnail {
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .thumbnail-image {
            width: 100%;
            max-height: 400px;
            height: auto;
            border-radius: 8px;
            object-fit: cover;
        }

        .article-title {
            font-size: 24pt;
            font-weight: bold;
            color: #2c5aa0;
            margin: 0 0 1rem 0;
            line-height: 1.2;
        }

        .article-description {
            font-size: 14pt;
            color: #666;
            margin: 0 0 1.5rem 0;
            font-style: italic;
        }


        /* Content Styles */
        .pdf-content {
            margin-bottom: 1.5rem;
        }

        .pdf-content h1 {
            font-size: 18pt;
            font-weight: bold;
            color: #2c5aa0;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }

        .pdf-content h2 {
            font-size: 16pt;
            font-weight: bold;
            color: #2c5aa0;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }

        .pdf-content h3 {
            font-size: 14pt;
            font-weight: bold;
            color: #2c5aa0;
            margin: 1.5rem 0 0.75rem 0;
            padding-top: 0.5rem;
        }

        .pdf-content h4 {
            font-size: 12pt;
            font-weight: bold;
            color: #444;
            margin: 1rem 0 0.5rem 0;
        }

        .pdf-content p {
            margin: 0 0 1rem 0;
            text-align: justify;
        }

        .pdf-content img {
            width: 100%;
            height: auto;
            display: block;
            margin: 1rem 0;
            border-radius: 4px;
            object-fit: cover;
        }

        .pdf-content ul, .pdf-content ol {
            margin: 0.75rem 0 1.25rem 0;
            padding-left: 1.5rem;
        }

        .pdf-content li {
            margin: 0.75rem 0;
            line-height: 1.6;
        }

        .pdf-content li:first-child {
            margin-top: 0.5rem;
        }

        .pdf-content li:last-child {
            margin-bottom: 0.5rem;
        }

        /* Issue and Suggestion Styling */
        .pdf-content li .issue-item {
            display: inline-block;
            color: #c62828;
            background-color: #ffebee;
            padding: 0.4rem 0.6rem;
            margin-left: 0.25rem;
            border-left: 3px solid #c62828;
            border-radius: 3px;
            font-weight: 500;
            width: calc(100% - 1rem);
        }

        .pdf-content li .suggestion-item {
            display: inline-block;
            color: #1565c0;
            background-color: #e3f2fd;
            padding: 0.4rem 0.6rem;
            margin-left: 0.25rem;
            border-left: 3px solid #1565c0;
            border-radius: 3px;
            font-weight: 500;
            width: calc(100% - 1rem);
        }

        .pdf-content li .strength-item {
            display: inline-block;
            color: #2e7d32;
            background-color: #e8f5e9;
            padding: 0.4rem 0.6rem;
            margin-left: 0.25rem;
            border-left: 3px solid #2e7d32;
            border-radius: 3px;
            width: calc(100% - 1rem);
        }

        .pdf-content li .quick-win-item {
            display: inline-block;
            color: #e65100;
            background-color: #fff3e0;
            padding: 0.4rem 0.6rem;
            margin-left: 0.25rem;
            border-left: 3px solid #e65100;
            border-radius: 3px;
            width: calc(100% - 1rem);
        }

        /* Section spacing improvements */
        .pdf-content h3 + h4 {
            margin-top: 0.75rem;
        }

        .pdf-content h4 + p,
        .pdf-content h4 + ul,
        .pdf-content h4 + ol {
            margin-top: 0.5rem;
        }

        .pdf-content blockquote {
            margin: 1rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-left: 4px solid #2c5aa0;
            font-style: italic;
        }

        .pdf-content code {
            background-color: #f1f3f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }

        .pdf-content pre {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .pdf-content pre code {
            background: none;
            padding: 0;
        }

        /* Table Styles */
        .pdf-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 10pt;
        }

        .pdf-content th,
        .pdf-content td {
            border: 1px solid #ddd;
            padding: 0.5rem;
            text-align: left;
        }

        .pdf-content th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c5aa0;
        }

        .pdf-content tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        /* Keywords Section */
        .keywords-section {
            margin: 2rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 5px;
        }

        .keywords-section h2 {
            font-size: 14pt;
            color: #2c5aa0;
            margin: 0 0 1rem 0;
        }

        .keywords-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .keyword-tag {
            background-color: #2c5aa0;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            font-size: 9pt;
        }

        /* FAQs Section */
        .faqs-section {
            margin: 2rem 0;
        }

        .faqs-section h2 {
            font-size: 16pt;
            color: #2c5aa0;
            margin: 0 0 1rem 0;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.5rem;
        }

        .faq-item {
            margin: 1rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 5px;
        }

        .faq-question {
            font-size: 12pt;
            font-weight: bold;
            color: #2c5aa0;
            margin: 0 0 0.5rem 0;
        }

        .faq-answer {
            font-size: 11pt;
            color: #555;
            margin: 0;
        }


        /* Page Break Utilities */
        .page-break {
            page-break-before: always;
        }

        .avoid-break {
            page-break-inside: avoid;
        }

        /* Print-specific styles */
        @media print {
            .pdf-content h1,
            .pdf-content h2,
            .pdf-content h3 {
                page-break-after: avoid;
            }
            
            .faq-item {
                page-break-inside: avoid;
            }
        }
        """
