// SiteOne Crawler - AI crawlers and fetchers known to the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Every row was checked against its vendor's own documentation on the date in `verified_on`.
// `note_key` names the report text that explains the row (EN/CS in the GEO document).

/// What a vendor uses a robots.txt product token for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Crawling for a search or answer engine index.
    Search,
    /// Fetching a page because a user asked the assistant to.
    UserFetch,
    /// Collecting content to train AI models.
    Training,
    /// Using crawled content to ground AI answers.
    Grounding,
}

/// Whether a vendor says its agent follows robots.txt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compliance {
    Honors,
    /// The vendor says robots.txt rules may not apply (user-initiated fetches).
    MayNotHonor,
    /// Not a crawler: a token the vendor reads in robots.txt to decide how it may use content
    /// fetched by its other crawlers.
    ControlToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiAgent {
    /// The robots.txt product token, as the vendor writes it.
    pub token: &'static str,
    pub vendor: &'static str,
    pub purposes: &'static [Purpose],
    pub compliance: Compliance,
    /// The token whose group the agent follows when robots.txt has no group for it but has
    /// one for this token.
    pub fallback_token: Option<&'static str>,
    pub note_key: &'static str,
    pub doc_url: &'static str,
    pub verified_on: &'static str,
}

impl AiAgent {
    pub fn is_training_only(&self) -> bool {
        self.purposes == [Purpose::Training]
    }
}

const GOOGLE_DOC: &str = "https://developers.google.com/crawling/docs/crawlers-fetchers/google-common-crawlers";
const BING_DOC: &str = "https://www.bing.com/webmasters/help/which-crawlers-does-bing-use-8c184ec0";
const OPENAI_DOC: &str = "https://developers.openai.com/api/docs/bots";
const ANTHROPIC_DOC: &str = "https://support.claude.com/en/articles/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler";
const PERPLEXITY_DOC: &str = "https://docs.perplexity.ai/docs/resources/perplexity-crawlers";
const APPLE_DOC: &str = "https://support.apple.com/en-us/119829";
const META_DOC: &str = "https://developers.facebook.com/documentation/sharing/webmasters/web-crawlers";
const COMMON_CRAWL_DOC: &str = "https://commoncrawl.org/ccbot";
const VERIFIED_ON: &str = "2026-09-26";

/// The AI-relevant crawlers and control tokens, in report order.
pub const AI_AGENTS: &[AiAgent] = &[
    // Google Search, including AI Overviews and AI Mode. A URL disallowed in robots.txt may still
    // be indexed and shown without a snippet.
    AiAgent {
        token: "Googlebot",
        vendor: "Google",
        purposes: &[Purpose::Search],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.googlebot",
        doc_url: GOOGLE_DOC,
        verified_on: VERIFIED_ON,
    },
    // Gemini training and grounding (Gemini Apps, Grounding with Google Search on Vertex AI);
    // no effect on Google Search or AI Overviews.
    AiAgent {
        token: "Google-Extended",
        vendor: "Google",
        purposes: &[Purpose::Training, Purpose::Grounding],
        compliance: Compliance::ControlToken,
        fallback_token: None,
        note_key: "agent.google_extended",
        doc_url: GOOGLE_DOC,
        verified_on: VERIFIED_ON,
    },
    // Bing search and Copilot share one index, also used for grounding. Use in Copilot answers
    // and in training is controlled by NOARCHIVE / NOCACHE, not by robots.txt.
    AiAgent {
        token: "Bingbot",
        vendor: "Microsoft",
        purposes: &[Purpose::Search, Purpose::Grounding],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.bingbot",
        doc_url: BING_DOC,
        verified_on: VERIFIED_ON,
    },
    // ChatGPT search. OpenAI also recommends allowing its published IP ranges.
    AiAgent {
        token: "OAI-SearchBot",
        vendor: "OpenAI",
        purposes: &[Purpose::Search],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.oai_searchbot",
        doc_url: OPENAI_DOC,
        verified_on: VERIFIED_ON,
    },
    // User actions in ChatGPT; "robots.txt rules may not apply".
    AiAgent {
        token: "ChatGPT-User",
        vendor: "OpenAI",
        purposes: &[Purpose::UserFetch],
        compliance: Compliance::MayNotHonor,
        fallback_token: None,
        note_key: "agent.chatgpt_user",
        doc_url: OPENAI_DOC,
        verified_on: VERIFIED_ON,
    },
    // Training of OpenAI's generative AI foundation models.
    AiAgent {
        token: "GPTBot",
        vendor: "OpenAI",
        purposes: &[Purpose::Training],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.gptbot",
        doc_url: OPENAI_DOC,
        verified_on: VERIFIED_ON,
    },
    // Claude search results.
    AiAgent {
        token: "Claude-SearchBot",
        vendor: "Anthropic",
        purposes: &[Purpose::Search],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.claude_searchbot",
        doc_url: ANTHROPIC_DOC,
        verified_on: VERIFIED_ON,
    },
    // Pages Claude fetches for a user's question; Anthropic honors robots.txt for them.
    AiAgent {
        token: "Claude-User",
        vendor: "Anthropic",
        purposes: &[Purpose::UserFetch],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.claude_user",
        doc_url: ANTHROPIC_DOC,
        verified_on: VERIFIED_ON,
    },
    // Training of Anthropic's models.
    AiAgent {
        token: "ClaudeBot",
        vendor: "Anthropic",
        purposes: &[Purpose::Training],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.claudebot",
        doc_url: ANTHROPIC_DOC,
        verified_on: VERIFIED_ON,
    },
    // Perplexity search results; not used for foundation models, and Perplexity lists no
    // training crawler.
    AiAgent {
        token: "PerplexityBot",
        vendor: "Perplexity",
        purposes: &[Purpose::Search],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.perplexitybot",
        doc_url: PERPLEXITY_DOC,
        verified_on: VERIFIED_ON,
    },
    // User actions in Perplexity; it "generally ignores robots.txt rules".
    AiAgent {
        token: "Perplexity-User",
        vendor: "Perplexity",
        purposes: &[Purpose::UserFetch],
        compliance: Compliance::MayNotHonor,
        fallback_token: None,
        note_key: "agent.perplexity_user",
        doc_url: PERPLEXITY_DOC,
        verified_on: VERIFIED_ON,
    },
    // Spotlight, Siri and Safari, and context for Apple's AI answers. Follows the Googlebot
    // group when robots.txt has one but no Applebot group.
    AiAgent {
        token: "Applebot",
        vendor: "Apple",
        purposes: &[Purpose::Search, Purpose::Grounding],
        compliance: Compliance::Honors,
        fallback_token: Some("Googlebot"),
        note_key: "agent.applebot",
        doc_url: APPLE_DOC,
        verified_on: VERIFIED_ON,
    },
    // Training of Apple's foundation models on Applebot's data; it does not crawl, and search
    // is unaffected.
    AiAgent {
        token: "Applebot-Extended",
        vendor: "Apple",
        purposes: &[Purpose::Training],
        compliance: Compliance::ControlToken,
        fallback_token: None,
        note_key: "agent.applebot_extended",
        doc_url: APPLE_DOC,
        verified_on: VERIFIED_ON,
    },
    // Training of Meta's AI models and product indexing.
    AiAgent {
        token: "meta-externalagent",
        vendor: "Meta",
        purposes: &[Purpose::Training],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.meta_externalagent",
        doc_url: META_DOC,
        verified_on: VERIFIED_ON,
    },
    // An open web archive whose data third parties widely use to train large language models
    // (https://commoncrawl.org/about).
    AiAgent {
        token: "CCBot",
        vendor: "Common Crawl",
        purposes: &[Purpose::Training],
        compliance: Compliance::Honors,
        fallback_token: None,
        note_key: "agent.ccbot",
        doc_url: COMMON_CRAWL_DOC,
        verified_on: VERIFIED_ON,
    },
];

/// The table row of a product token (case-insensitive).
pub fn find_agent(token: &str) -> Option<&'static AiAgent> {
    AI_AGENTS.iter().find(|agent| agent.token.eq_ignore_ascii_case(token))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(token: &str) -> &'static AiAgent {
        find_agent(token).unwrap_or_else(|| panic!("{token} is in the table"))
    }

    #[test]
    fn the_table_holds_every_verified_agent_once() {
        let tokens: Vec<&str> = AI_AGENTS.iter().map(|agent| agent.token).collect();
        assert_eq!(
            tokens,
            [
                "Googlebot",
                "Google-Extended",
                "Bingbot",
                "OAI-SearchBot",
                "ChatGPT-User",
                "GPTBot",
                "Claude-SearchBot",
                "Claude-User",
                "ClaudeBot",
                "PerplexityBot",
                "Perplexity-User",
                "Applebot",
                "Applebot-Extended",
                "meta-externalagent",
                "CCBot",
            ]
        );
        for agent in AI_AGENTS {
            // RFC 9309 §2.2.1: a product token has only letters, underscores and hyphens.
            assert!(
                agent
                    .token
                    .bytes()
                    .all(|b| b.is_ascii_alphabetic() || b == b'_' || b == b'-'),
                "{}",
                agent.token
            );
            assert!(
                !agent.vendor.is_empty() && !agent.purposes.is_empty(),
                "{}",
                agent.token
            );
            assert!(agent.doc_url.starts_with("https://"), "{}", agent.token);
            assert!(!agent.note_key.is_empty(), "{}", agent.token);
            assert!(
                chrono::NaiveDate::parse_from_str(agent.verified_on, "%Y-%m-%d").is_ok(),
                "{}: {}",
                agent.token,
                agent.verified_on
            );
        }
    }

    #[test]
    fn agents_are_found_by_a_case_insensitive_token() {
        assert_eq!(agent("oai-searchbot").token, "OAI-SearchBot");
        assert_eq!(agent("Meta-ExternalAgent").token, "meta-externalagent");
        assert!(find_agent("SiteOne-Crawler").is_none());
    }

    #[test]
    fn roles_follow_the_vendor_documentation() {
        use Compliance::*;
        use Purpose::*;

        let expected: [(&str, &[Purpose], Compliance); 15] = [
            ("Googlebot", &[Search], Honors),
            ("Google-Extended", &[Training, Grounding], ControlToken),
            ("Bingbot", &[Search, Grounding], Honors),
            ("OAI-SearchBot", &[Search], Honors),
            ("ChatGPT-User", &[UserFetch], MayNotHonor),
            ("GPTBot", &[Training], Honors),
            ("Claude-SearchBot", &[Search], Honors),
            ("Claude-User", &[UserFetch], Honors),
            ("ClaudeBot", &[Training], Honors),
            ("PerplexityBot", &[Search], Honors),
            ("Perplexity-User", &[UserFetch], MayNotHonor),
            ("Applebot", &[Search, Grounding], Honors),
            ("Applebot-Extended", &[Training], ControlToken),
            ("meta-externalagent", &[Training], Honors),
            ("CCBot", &[Training], Honors),
        ];
        for (token, purposes, compliance) in expected {
            let agent = agent(token);
            assert_eq!(agent.purposes, purposes, "{token}");
            assert_eq!(agent.compliance, compliance, "{token}");
        }
    }

    #[test]
    fn only_applebot_falls_back_to_another_token() {
        for agent in AI_AGENTS {
            let expected = (agent.token == "Applebot").then_some("Googlebot");
            assert_eq!(agent.fallback_token, expected, "{}", agent.token);
        }
    }

    #[test]
    fn training_only_means_no_other_purpose() {
        assert!(agent("GPTBot").is_training_only());
        assert!(agent("Applebot-Extended").is_training_only());
        assert!(!agent("Google-Extended").is_training_only());
        assert!(!agent("Googlebot").is_training_only());
    }
}
