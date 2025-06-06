# SiteOne Crawler: Text Output Documentation

## Table of Contents

*   [1. Introduction](#1-introduction)
*   [2. General Format](#2-general-format)
*   [3. Detailed Section Breakdown](#3-detailed-section-breakdown)
    *   [3.1. Progress Report](#31-progress-report)
    *   [3.2. Skipped URLs Summary](#32-skipped-urls-summary)
    *   [3.3. Skipped URLs](#33-skipped-urls)
    *   [3.4. Redirected URLs](#34-redirected-urls)
    *   [3.5. 404 URLs](#35-404-urls)
    *   [3.6. SSL/TLS Info](#36-ssltls-info)
    *   [3.7. Performance Metrics (Fastest/Slowest URLs)](#37-performance-metrics-fastestslowest-urls)
    *   [3.8. SEO & Content Analysis](#38-seo--content-analysis)
    *   [3.9. HTTP Headers](#39-http-headers)
    *   [3.10. HTTP Caching](#310-http-caching)
    *   [3.11. Best Practices](#311-best-practices)
    *   [3.12. Accessibility](#312-accessibility)
    *   [3.13. Source Domains](#313-source-domains)
    *   [3.14. Content Types](#314-content-types)
    *   [3.15. DNS Info](#315-dns-info)
    *   [3.16. Security](#316-security)
    *   [3.17. Analysis Stats](#317-analysis-stats)
*   [4. Information Obtainable from Text Output](#4-information-obtainable-from-text-output)
*   [5. Use Cases for Text Output](#5-use-cases-for-text-output)
*   [6. Note on JSON Output](#6-note-on-json-output)


This document describes the format of the text (`.txt`) output generated by the SiteOne Crawler tool. This output provides a comprehensive summary of the crawl results in a human-readable format, suitable for quick analysis and review directly in a text editor or terminal.

## 1. Introduction

The text output begins with an ASCII art logo, version information, and the author's contact details. This is followed by several sections detailing various aspects of the crawled website. The primary sections include:

*   **Progress Report:** Real-time status of crawled URLs.
*   **Skipped URLs Summary:** Aggregated counts of URLs skipped for various reasons.
*   **Skipped URLs:** Detailed list of skipped URLs, reasons, and sources.
*   **Redirected URLs:** List of URLs that resulted in redirects.
*   **404 URLs:** List of URLs that returned a 404 Not Found status.
*   **SSL/TLS Info:** Details about the website's SSL/TLS certificate.
*   **Performance Metrics:** Top fastest and slowest URLs.
*   **SEO &amp; Content Analysis:** SEO metadata, OpenGraph metadata, heading structure, and duplicate content reports.
*   **HTTP Headers:** Analysis of HTTP headers found during the crawl.
*   **HTTP Caching:** Detailed breakdown of caching strategies by content type and domain.
*   **Best Practices:** Results of various best practice checks.
*   **Accessibility:** Results of accessibility checks.
*   **Source Domains:** Summary of crawled domains.
*   **Content Types:** Summary of crawled content types (general and MIME types).
*   **DNS Info:** Information about DNS resolution.
*   **Security:** Results of security header checks.
*   **Analysis Stats:** Performance statistics for the crawler's internal analyzers.

## 2. General Format

The output uses simple text formatting:

*   **Headers:** Section titles are typically preceded by `---` lines and followed by `===` or `---` lines for visual separation.
*   **Tables:** Data is presented in fixed-width tables with headers underlined by hyphens (`-`). Column alignment is maintained using spaces. This documentation uses Markdown tables for examples.
*   **Truncation:** Some tables containing potentially large amounts of data (like SEO metadata or heading structures) might show only a limited number of rows (e.g., max 10) in the text output, with a note advising the use of the HTML report (`--output-html-report`) for the complete data.

## 3. Detailed Section Breakdown

### 3.1. Progress Report

This section shows the progress of the crawl in real-time (or the final state if the crawl is complete).

| Progress | %    | Bar      | URL                                                     | Status | Type | Time  | Size  | Cache  | Access. | Best pr. |
| :------- | :--- | :------- | :------------------------------------------------------ | :----- | :--- | :---- | :---- | :----- | :------ | :------- |
| 1/43     | 2%   |          | /                                                       | 200    | HTML | 34 ms | 45 kB | 60 min | 1/1/2   | 1/5      |
| 2/64     | 3%   |          | /installation-and-requirements/ready-to-use-packages/ | 200    | HTML | 13 ms | 61 kB | 60 min | 1/3     | 1/5      |
| ...      | ...  | ...      | ...                                                     | ...    | ...  | ...   | ...   | ...    | ...     | ...      |

*   **Progress report:** Columns include:
    *   `Progress` (`X/Y`): `X` = URL sequence number, `Y` = Total URLs found so far.
    *   `%`: Percentage of URLs processed relative to the total found.
    *   `Bar`: Visual indicator (`>`, `>>`, etc.).
    *   `URL`: The path or full URL being processed.
    *   `Status`: HTTP status code returned (e.g., 200, 404, 301).
    *   `Type`: Detected content type (e.g., HTML, JS, CSS, Image).
    *   `Time`: Time taken to download the URL.
    *   `Size`: Size of the downloaded content.
    *   `Cache`: Detected cache lifetime (e.g., 60 min, 12 mon, none).
    *   `Access.`: Accessibility issues summary (OK/Notice/Warning/Critical).
    *   `Best pr.`: Best practices issues summary (OK/Notice/Warning/Critical).

### 3.2. Skipped URLs Summary

Provides a high-level overview of why URLs were skipped during the crawl, grouped by reason and domain.

**Skipped URLs Summary**

| Reason           | Domain             | Unique URLs |
| :--------------- | :----------------- | :---------- |
| Not allowed host | nextjs.org         | 1294        |
| Not allowed host | astro.build        | 925         |
| Robots.txt       | crawler.siteone.io | 3           |
| ...              | ...                | ...         |

*   **Reason:** Why the URL was skipped (e.g., `Not allowed host`, `Robots.txt`, `Max depth reached`).
*   **Domain:** The domain of the skipped URLs.
*   **Unique URLs:** The count of unique URLs skipped for that reason/domain combination.

### 3.3. Skipped URLs

Lists individual skipped URLs with more context.

**Skipped URLs**

| Reason           | Skipped URL           | Source   | Found at URL                                         |
| :--------------- | :-------------------- | :------- | :--------------------------------------------------- |
| Not allowed host | http://astro.build/   | `<a href>` | /html/2024-08-24/forever/hwzxj1-qrs69-1fqlxbd.html |
| Not allowed host | https://adamwathan.me/ | `<a href>` | /introduction/thanks/                                |
| ...              | ...                   | ...      | ...                                                  |

*   **Reason:** Why the URL was skipped.
*   **Skipped URL:** The specific URL that was not crawled.
*   **Source:** How the URL was discovered (e.g., `<a href>`, `<img src>`, `CSS url()`).
*   **Found at URL:** The URL where the skipped URL was found.

### 3.4. Redirected URLs

Lists URLs that resulted in an HTTP redirect. (Empty in the example, but would follow a similar table format if redirects were found).

### 3.5. 404 URLs

Lists URLs that returned a 404 Not Found status code.

**404 URLs**

| Status | URL 404                                                             | Found at URL                                                              |
| :----- | :------------------------------------------------------------------ | :------------------------------------------------------------------------ |
| 404    | https://crawler.siteone.io/html/2024-08-23/forever/httpAgentOptions | https://crawler.siteone.io/html/2024-08-23/forever/cl8xw4r-fdag8wg-44dd.html |
| ...    | ...                                                                 | ...                                                                       |

*   **Status:** The HTTP status code (typically 404).
*   **URL 404:** The URL that resulted in the 404 error.
*   **Found at URL:** The URL containing the link to the broken page.

### 3.6. SSL/TLS Info

Provides details about the SSL/TLS certificate of the primary host.

**SSL/TLS info**

| Info                   | Text                                                                    |
| :--------------------- | :---------------------------------------------------------------------- |
| Issuer                 | C = BE, O = GlobalSign nv-sa, CN = GlobalSign GCC R6 AlphaSSL CA 2023 |
| Subject                | CN = *.siteone.io                                                       |
| Valid from             | Jan 23 09:52:19 2025 GMT (VALID already 73.2 day(s))                    |
| Valid to               | Feb 24 09:52:18 2026 GMT (VALID still for 323.8 day(s))                 |
| Supported protocols    | TLSv1.2                                                                 |
| RAW certificate output | `Certificate: ...` (details omitted)                                    |
| RAW protocols output   | `=== ssl2 === ...` (details omitted)                                    |

*   **Info:** The type of information (Issuer, Subject, Validity dates, Supported protocols).
*   **Text:** The corresponding value for the information type. Includes raw output snippets.

### 3.7. Performance Metrics (Fastest/Slowest URLs)

Two tables listing the top N fastest and slowest URLs encountered during the crawl.

**TOP fastest URLs**

| Time  | Status | Fast URL                                                              |
| :---- | :----- | :-------------------------------------------------------------------- |
| 11 ms | 200    | https://crawler.siteone.io/installation-and-requirements/desktop-application/ |
| ...   | ...    | ...                                                                   |

**TOP slowest URLs**

| Time  | Status | Slow URL                                                              |
| :---- | :----- | :-------------------------------------------------------------------- |
| 2.7 s | 200    | https://crawler.siteone.io/html/2024-08-24/forever/hwzxj1-qrs69-1fqlxbd.html |
| ...   | ...    | ...                                                                   |

*   **Time:** Time taken to download the URL.
*   **Status:** HTTP status code.
*   **Fast/Slow URL:** The URL itself.

### 3.8. SEO & Content Analysis

Includes several sub-sections like SEO metadata, OpenGraph metadata, heading structure, and duplicate content reports. *(Note: These tables are often truncated in the text output and are not shown here in Markdown format for brevity).*

### 3.9. HTTP Headers

Analyzes HTTP response headers across all crawled URLs.

*   **HTTP headers (Summary):** Lists unique headers, occurrence count, unique value count, preview of values, and min/max values where applicable (e.g., for Content-Length or dates).
*   **HTTP header values (Detailed):** Lists specific values for headers with multiple distinct values, along with their occurrence counts.

**HTTP headers (Summary Example)**

| Header        | Occurs | Unique | Values preview                           | Min value | Max value |
| :------------ | :----- | :----- | :--------------------------------------- | :-------- | :-------- |
| Accept-Ranges | 12     | 1      | bytes                                    |           |           |
| Cache-Control | 66     | 2      | max-age=3600 (49) / max-age=31536000 (17) |           |           |
| ...           | ...    | ...    | ...                                      | ...       | ...       |

**HTTP header values (Detailed Example)**

| Header        | Occurs | Value        |
| :------------ | :----- | :----------- |
| Accept-Ranges | 12     | bytes        |
| Cache-Control | 49     | max-age=3600 |
| ...           | ...    | ...          |

### 3.10. HTTP Caching

Provides detailed analysis of HTTP caching headers.

*   **HTTP Caching by content type:** Summarizes caching strategies (e.g., `Cache-Control + ETag`, `No cache headers`) used for different content types (HTML, CSS, JS, Image, etc.), including counts and average/min/max lifetimes.
*   **HTTP Caching by domain:** Similar summary, but grouped by domain.
*   **HTTP Caching by domain and content type:** The most granular view, showing caching strategies for each content type within each domain.

**HTTP Caching by content type (Example)**

| Content type | Cache type                           | URLs | AVG lifetime | MIN lifetime | MAX lifetime |
| :----------- | :----------------------------------- | :--- | :----------- | :----------- | :----------- |
| HTML         | Cache-Control + ETag + Last-Modified | 45   | 60 min       | 60 min       | 60 min       |
| Image        | Cache-Control + ETag + Last-Modified | 11   | 12 mon       | 12 mon       | 12 mon       |
| ...          | ...                                  | ...  | ...          | ...          | ...          |

### 3.11. Best Practices

Summarizes results from various best practice checks.

**Best practices (Example)**

| Analysis name                            | OK  | Notice | Warning | Critical |
| :--------------------------------------- | :-- | :----- | :------ | :------- |
| Large inline SVGs (> 5120 B)             | 148 | 0      | 108     | 0        |
| Invalid inline SVGs                      | 63  | 0      | 193     | 0        |
| Heading structure                        | 0   | 47     | 0       | 0        |
| ...                                      | ... | ...    | ...     | ...      |

*   **Analysis name:** The specific check performed.
*   **OK / Notice / Warning / Critical:** Counts of URLs falling into each severity category for that check.

### 3.12. Accessibility

Summarizes results from accessibility checks.

**Accessibility (Example)**

| Analysis name                | OK   | Notice | Warning | Critical |
| :--------------------------- | :--- | :----- | :------ | :------- |
| Missing image alt attributes | 1419 | 0      | 2       | 0        |
| Missing html lang attribute  | 0    | 0      | 0       | 1        |
| ...                          | ...  | ...    | ...     | ...      |

*   **Analysis name:** The specific accessibility check.
*   **OK / Notice / Warning / Critical:** Counts for each severity level.

### 3.13. Source Domains

Lists all domains from which resources were successfully crawled, with counts and size/time summaries per content type.

**Source domains (Example)**

| Domain             | Totals       | HTML       | Image      | JS          | CSS         | Document    | JSON      |
| :----------------- | :----------- | :--------- | :--------- | :---------- | :---------- | :---------- | :-------- |
| crawler.siteone.io | 67/30MB/6.2s | 48/12MB/4s | 11/18MB/2s | 4/13kB/54ms | 2/77kB/41ms | 1/135B/10ms | 1/36B/14ms |

### 3.14. Content Types

Summarizes crawled resources by content type.

*   **Content types (General):** Groups by broad categories (HTML, Image, JS, CSS, etc.).
*   **Content types (MIME types):** Groups by specific MIME types (e.g., `text/html`, `image/jpeg`, `application/javascript`).

**Content types (General Example)**

| Content type | URLs | Total size | Total time | Avg time | Status 20x | Status 40x |
| :----------- | :--- | :--------- | :--------- | :------- | :--------- | :--------- |
| HTML         | 48   | 12 MB      | 4 s        | 84 ms    | 48         | 0          |
| Image        | 11   | 18 MB      | 2 s        | 185 ms   | 11         | 0          |
| ...          | ...  | ...        | ...        | ...      | ...        | ...        |

**Content types (MIME types Example)**

| Content type           | URLs | Total size | Total time | Avg time | Status 20x | Status 40x |
| :--------------------- | :--- | :--------- | :--------- | :------- | :--------- | :--------- |
| text/html              | 45   | 2 MB       | 1.2 s      | 26 ms    | 45         | 0          |
| application/javascript | 4    | 13 kB      | 54 ms      | 14 ms    | 4          | 0          |
| ...                    | ...  | ...        | ...        | ...      | ...        | ...        |

### 3.15. DNS Info

Shows the DNS resolution tree for the crawled domain(s) and the DNS server used. (Not a table format).

```
DNS info
--------

DNS resolving tree                                                    
------------------------------------------------------------------------
crawler.siteone.io                                                    
  IPv4: 86.49.167.242                                                 
                                                                      
DNS server: 10.255.255.254                                            
```

### 3.16. Security

Reports on the presence and configuration of important security-related HTTP headers.

**Security (Example)**

| Header                    | OK | Notice | Warning | Critical | Recommendation                                                                                             |
| :------------------------ | :- | :----- | :------ | :------- | :--------------------------------------------------------------------------------------------------------- |
| Strict-Transport-Security | 45 | 0      | 0       | 3        | Strict-Transport-Security header is not set. It enforces secure connections and protects against MITM attacks. |
| X-XSS-Protection          | 45 | 0      | 0       | 3        | X-XSS-Protection header is not set. It enables browser's built-in defenses against XSS attacks.            |
| ...                       | .. | ...    | ...     | ...      | ...                                                                                                        |

*   **Header:** The security header being checked.
*   **OK / Notice / Warning / Critical:** Counts based on the header's presence and configuration.
*   **Recommendation:** Suggestion for improvement if issues are found.

### 3.17. Analysis Stats

Provides performance metrics for the crawler's internal analysis modules. Useful for debugging the crawler itself.

**Analysis stats (Example)**

| Class::method                               | Exec time | Exec count |
| :------------------------------------------ | :-------- | :--------- |
| Manager::parseDOMDocument                   | 707 ms    | 48         |
| SslTlsAnalyzer::getTLSandSSLCertificateInfo | 215 ms    | 1          |
| ...                                         | ...       | ...        |

## 4. Information Obtainable from Text Output

The text output provides a wealth of information about a website, including:

*   **Crawl Overview:** Number of pages found, processed, and skipped.
*   **Website Structure:** Implicitly through the list of crawled URLs and their relationships (via "Found at URL").
*   **Link Health:** Identification of broken links (404s) and redirects.
*   **External Dependencies:** List of external domains linked to or hosting resources (from Skipped URLs).
*   **Performance Bottlenecks:** Identification of the slowest loading pages and resources.
*   **Content Inventory:** Summary of different content types (HTML, images, scripts, stylesheets) and their sizes/load times.
*   **Basic SEO Health:** Status of titles, descriptions, heading structures, and indexing directives.
*   **OpenGraph Implementation:** Presence and content of OG tags for social sharing.
*   **Server Configuration:** Insights into HTTP headers used, including caching and security headers.
*   **Caching Strategy:** Effectiveness of caching policies across different content types and domains.
*   **Security Posture:** Checks for essential security headers (HSTS, X-Frame-Options, etc.).
*   **Accessibility Issues:** High-level view of common accessibility problems (missing alt text, lang attributes).
*   **Best Practice Adherence:** Checks against common web development best practices.
*   **SSL/TLS Certificate Status:** Validity and issuer details of the site's certificate.

## 5. Use Cases for Text Output

The text output is valuable for various tasks:

1.  **Quick Website Health Check:** Get a fast overview of major issues like 404s, slow pages, or critical security/accessibility warnings.
2.  **Identifying Broken Links:** Easily spot and locate 404 errors using the dedicated section.
3.  **Performance Audit:** Identify the slowest URLs to prioritize optimization efforts.
4.  **Basic SEO Audit:** Check for duplicate titles/descriptions and analyze heading structures.
5.  **Security Header Review:** Quickly verify the presence of important security headers.
6.  **Caching Policy Verification:** Understand how caching is implemented across the site.
7.  **Pre/Post Deployment Checks:** Compare outputs before and after changes to catch regressions.
8.  **Generating Simple Reports:** Copy-paste relevant sections into emails or documents for concise reporting.
9.  **Troubleshooting Crawl Issues:** Use skipped URLs and analysis stats to understand crawler behavior.
10. **Command-Line Integration:** Process the text output with standard command-line tools (grep, awk, sed) for specific data extraction or automated checks in simple scripts.

## 6. Note on JSON Output

While this document focuses on the text output, SiteOne Crawler also offers a JSON output format (`--output-json-file`). The JSON output contains much of the same information but in a structured format that is ideal for programmatic consumption, detailed data analysis, or integration with other tools and dashboards. For automated processing or complex data manipulation, the JSON output is generally preferred.

See the [JSON Output Documentation](JSON-OUTPUT.md) for more details on the JSON format.