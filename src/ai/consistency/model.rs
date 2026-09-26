// SiteOne Crawler - AI fact consistency: data model
// (c) Jan Reges <jan.reges@siteone.cz>
//
// The facts the pipeline passes between its steps: the sources shown to the model, the facts it
// cited, the crawler-verified occurrences, the fact keys they are grouped into, and the review
// outcome of each group.

use crate::ai::grounding::{ValueHint, ValueKey};

/// The kind of property a fact states, from a controlled vocabulary: the extractor must pick one
/// (anything unknown becomes `Other`), and grouping compares facts only within one key, so a
/// free-shipping threshold is one bucket whether the model called it a price or a condition, and
/// an interest rate never meets an APR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeKey {
    // contact
    Phone,
    Email,
    PostalAddress,
    OpeningHours,
    ContactPerson,
    // money
    Price,
    Fee,
    InterestRate,
    Apr,
    Discount,
    Commission,
    Deposit,
    AmountLimit,
    // terms
    FreeShippingThreshold,
    MinimumOrder,
    DeliveryTime,
    ReturnPeriod,
    WarrantyPeriod,
    CancellationPeriod,
    AgeLimit,
    Eligibility,
    // time
    Deadline,
    Validity,
    EventDate,
    FoundedYear,
    // figures
    CustomersCount,
    EmployeesCount,
    BranchesCount,
    YearsInBusiness,
    Rating,
    Spec,
    OtherFigure,
    // identity
    CompanyId,
    VatId,
    BankAccount,
    RegistrationNumber,
    LicenceNumber,
    PersonRole,
    Other,
}

impl AttributeKey {
    /// The whole vocabulary, in declaration order.
    pub const ALL: [AttributeKey; 39] = {
        use AttributeKey::*;
        [
            Phone,
            Email,
            PostalAddress,
            OpeningHours,
            ContactPerson,
            Price,
            Fee,
            InterestRate,
            Apr,
            Discount,
            Commission,
            Deposit,
            AmountLimit,
            FreeShippingThreshold,
            MinimumOrder,
            DeliveryTime,
            ReturnPeriod,
            WarrantyPeriod,
            CancellationPeriod,
            AgeLimit,
            Eligibility,
            Deadline,
            Validity,
            EventDate,
            FoundedYear,
            CustomersCount,
            EmployeesCount,
            BranchesCount,
            YearsInBusiness,
            Rating,
            Spec,
            OtherFigure,
            CompanyId,
            VatId,
            BankAccount,
            RegistrationNumber,
            LicenceNumber,
            PersonRole,
            Other,
        ]
    };

    /// The vocabulary name (as in the prompts and the JSON).
    pub fn as_str(self) -> &'static str {
        use AttributeKey::*;
        match self {
            Phone => "phone",
            Email => "email",
            PostalAddress => "postal_address",
            OpeningHours => "opening_hours",
            ContactPerson => "contact_person",
            Price => "price",
            Fee => "fee",
            InterestRate => "interest_rate",
            Apr => "apr",
            Discount => "discount",
            Commission => "commission",
            Deposit => "deposit",
            AmountLimit => "amount_limit",
            FreeShippingThreshold => "free_shipping_threshold",
            MinimumOrder => "minimum_order",
            DeliveryTime => "delivery_time",
            ReturnPeriod => "return_period",
            WarrantyPeriod => "warranty_period",
            CancellationPeriod => "cancellation_period",
            AgeLimit => "age_limit",
            Eligibility => "eligibility",
            Deadline => "deadline",
            Validity => "validity",
            EventDate => "event_date",
            FoundedYear => "founded_year",
            CustomersCount => "customers_count",
            EmployeesCount => "employees_count",
            BranchesCount => "branches_count",
            YearsInBusiness => "years_in_business",
            Rating => "rating",
            Spec => "spec",
            OtherFigure => "other_figure",
            CompanyId => "company_id",
            VatId => "vat_id",
            BankAccount => "bank_account",
            RegistrationNumber => "registration_number",
            LicenceNumber => "licence_number",
            PersonRole => "person_role",
            Other => "other",
        }
    }

    /// The key a model named, tolerating case, spaces and hyphens; anything else is `Other`.
    pub fn parse(s: &str) -> Self {
        let wanted: String = s
            .trim()
            .chars()
            .map(|c| {
                if c == ' ' || c == '-' {
                    '_'
                } else {
                    c.to_ascii_lowercase()
                }
            })
            .collect();
        Self::ALL
            .into_iter()
            .find(|key| key.as_str() == wanted)
            .unwrap_or(AttributeKey::Other)
    }

    /// How a value of this key is read into a comparison key.
    pub fn value_hint(self) -> ValueHint {
        use AttributeKey::*;
        match self {
            Phone | Email => ValueHint::Contact,
            Price
            | Fee
            | InterestRate
            | Apr
            | Discount
            | Commission
            | Deposit
            | AmountLimit
            | FreeShippingThreshold
            | MinimumOrder
            | DeliveryTime
            | ReturnPeriod
            | WarrantyPeriod
            | CancellationPeriod
            | AgeLimit
            | FoundedYear
            | CustomersCount
            | EmployeesCount
            | BranchesCount
            | YearsInBusiness
            | Rating
            | Spec
            | OtherFigure => ValueHint::Number,
            Deadline | Validity | EventDate => ValueHint::Date,
            PostalAddress | OpeningHours | ContactPerson | Eligibility | CompanyId | VatId | BankAccount
            | RegistrationNumber | LicenceNumber | PersonRole | Other => ValueHint::Text,
        }
    }

    /// How much a wrong value could matter to a visitor, for sorting and for the review
    /// allocation: money and identity 5; contact and terms 4; time 3; figures 2; other 1.
    pub fn importance(self) -> u8 {
        use AttributeKey::*;
        match self {
            Price | Fee | InterestRate | Apr | Discount | Commission | Deposit | AmountLimit | CompanyId | VatId
            | BankAccount | RegistrationNumber | LicenceNumber | PersonRole => 5,
            Phone
            | Email
            | PostalAddress
            | OpeningHours
            | ContactPerson
            | FreeShippingThreshold
            | MinimumOrder
            | DeliveryTime
            | ReturnPeriod
            | WarrantyPeriod
            | CancellationPeriod
            | AgeLimit
            | Eligibility => 4,
            Deadline | Validity | EventDate | FoundedYear => 3,
            CustomersCount | EmployeesCount | BranchesCount | YearsInBusiness | Rating | Spec | OtherFigure => 2,
            Other => 1,
        }
    }

    /// Whether a difference can be `critical`: only where relying on the wrong value could
    /// directly cost a customer money or legal certainty.
    pub fn critical_allowed(self) -> bool {
        use AttributeKey::*;
        matches!(
            self,
            Price | Fee | InterestRate | Apr | CompanyId | VatId | BankAccount | RegistrationNumber
        )
    }
}

/// Where a source comes from: one page's own content, or header/footer lines shared by pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Page,
    Chrome,
}

/// One numbered block of a source, as the model sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceBlock {
    /// The crawler-made id the model cites: `B12` for a page block, `L7` for a header/footer line.
    pub ref_id: String,
    /// The crawler's copy of the text the model was shown (a block cut for length ends with the
    /// crawler's truncation note).
    pub text: String,
    /// The enclosing headings of a page block; for a header/footer line, its context label (the
    /// nearest preceding line without a fact), or nothing.
    pub heading_path: Vec<String>,
    /// The indexes of the pages showing this text: the page itself for a page block, the exact
    /// page set of a header/footer line.
    pub pages: Vec<usize>,
}

/// What one extraction call analyzes: a page's main content, or a chunk of header/footer lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisSource {
    /// A page source has its page's index; header/footer chunks follow after the pages.
    pub id: usize,
    pub kind: SourceKind,
    /// The page URL, or "" for header/footer lines.
    pub url: String,
    /// The page path, or `header/footer lines {i}/{n}`.
    pub path: String,
    pub blocks: Vec<SourceBlock>,
    /// Blocks of the page left out to fit the input budget (never inspected).
    pub omitted_blocks: usize,
    /// Included blocks cut for length (only their beginning was inspected).
    pub truncated_blocks: usize,
}

/// A page selected for the analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub index: usize,
    pub url: String,
    pub path: String,
    pub title: String,
}

/// One fact as the model returned it, before any verification.
#[derive(Debug, Clone, serde::Deserialize, Default, PartialEq, Eq)]
pub struct RawFact {
    #[serde(default)]
    pub block: String,
    #[serde(default)]
    pub attribute_key: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub attribute: String,
    #[serde(default)]
    pub value: String,
    #[serde(default)]
    pub qualifiers: String,
    #[serde(default)]
    pub quote: String,
    #[serde(default)]
    pub normalized: String,
}

/// A fact the crawler verified against the block it cites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occurrence {
    pub id: usize,
    /// The id of the `AnalysisSource`.
    pub source: usize,
    /// Page content or a header/footer line.
    pub region: SourceKind,
    /// The cited block (`B12` / `L7`).
    pub block_ref: String,
    pub attribute_key: AttributeKey,
    pub subject: String,
    pub attribute: String,
    /// The value as the page writes it (the crawler's copy).
    pub value: String,
    pub value_key: ValueKey,
    pub qualifiers: String,
    /// The crawler's text around the value, from the cited block.
    pub evidence: String,
    /// The byte range of the value within `evidence` (the cited copy of a repeated value).
    pub value_span: (usize, usize),
    pub heading_path: Vec<String>,
    /// The pages showing this occurrence.
    pub pages: Vec<usize>,
}

/// Facts about the same property of the same subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactKey {
    pub id: usize,
    pub attribute_key: AttributeKey,
    pub name: String,
    pub aliases: Vec<String>,
    pub occurrence_ids: Vec<usize>,
}

/// A key whose occurrences state one exact value in several places.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsistentFact {
    pub key_id: usize,
    pub name: String,
    pub attribute_key: AttributeKey,
    /// The value as the pages write it (the most common spelling).
    pub value: String,
    /// The distinct places stating it: pages, and header/footer lines.
    pub sources: usize,
    pub pages: usize,
    pub occurrence_ids: Vec<usize>,
}

/// How important a finding is to check. The declaration order is the sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// How sure the review is that the values disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Likely,
    Possible,
}

/// What became of a reviewed group (or of a subset of its values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Finding,
    Explainable,
    NotComparable,
    InsufficientContext,
    NotReviewed(NotReviewedReason),
}

/// Why a group was not reviewed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NotReviewedReason {
    /// Over the review cap.
    Cap,
    /// The review call failed or gave no valid result for the group.
    CallFailed,
    /// `--ai-max-tokens` is too low for a review.
    OutputBudget,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_key_parses_the_vocabulary_and_falls_back_to_other() {
        assert_eq!(AttributeKey::ALL.len(), 39);
        for key in AttributeKey::ALL {
            assert_eq!(AttributeKey::parse(key.as_str()), key, "{key:?}");
            assert_eq!(
                serde_json::to_value(key).unwrap(),
                serde_json::json!(key.as_str()),
                "the JSON name is the vocabulary name"
            );
        }
        assert_eq!(AttributeKey::parse(" APR "), AttributeKey::Apr);
        assert_eq!(AttributeKey::parse("Interest-Rate"), AttributeKey::InterestRate);
        assert_eq!(
            AttributeKey::parse("free shipping threshold"),
            AttributeKey::FreeShippingThreshold
        );
        assert_eq!(AttributeKey::parse("vat_id"), AttributeKey::VatId);
        for unknown in ["", "price_per_month", "cena", "</attribute_key>", "other_figures"] {
            assert_eq!(AttributeKey::parse(unknown), AttributeKey::Other, "{unknown:?}");
        }
    }

    #[test]
    fn importance_follows_the_attribute_families() {
        use AttributeKey::*;
        for key in [
            Price,
            Fee,
            InterestRate,
            Apr,
            Discount,
            Commission,
            Deposit,
            AmountLimit,
        ] {
            assert_eq!(key.importance(), 5, "money {key:?}");
        }
        for key in [
            CompanyId,
            VatId,
            BankAccount,
            RegistrationNumber,
            LicenceNumber,
            PersonRole,
        ] {
            assert_eq!(key.importance(), 5, "identity {key:?}");
        }
        for key in [Phone, Email, PostalAddress, OpeningHours, ContactPerson] {
            assert_eq!(key.importance(), 4, "contact {key:?}");
        }
        for key in [
            FreeShippingThreshold,
            MinimumOrder,
            DeliveryTime,
            ReturnPeriod,
            WarrantyPeriod,
            CancellationPeriod,
            AgeLimit,
            Eligibility,
        ] {
            assert_eq!(key.importance(), 4, "terms {key:?}");
        }
        for key in [Deadline, Validity, EventDate, FoundedYear] {
            assert_eq!(key.importance(), 3, "time {key:?}");
        }
        for key in [
            CustomersCount,
            EmployeesCount,
            BranchesCount,
            YearsInBusiness,
            Rating,
            Spec,
            OtherFigure,
        ] {
            assert_eq!(key.importance(), 2, "figures {key:?}");
        }
        assert_eq!(Other.importance(), 1);
    }

    #[test]
    fn critical_is_allowed_only_for_money_and_legal_identity() {
        use AttributeKey::*;
        let allowed: Vec<AttributeKey> = AttributeKey::ALL
            .into_iter()
            .filter(|key| key.critical_allowed())
            .collect();
        assert_eq!(
            allowed,
            vec![
                Price,
                Fee,
                InterestRate,
                Apr,
                CompanyId,
                VatId,
                BankAccount,
                RegistrationNumber
            ]
        );
    }

    #[test]
    fn value_hints_follow_the_kind_of_value() {
        use AttributeKey::*;
        assert_eq!(Phone.value_hint(), ValueHint::Contact);
        assert_eq!(Email.value_hint(), ValueHint::Contact);
        for key in [
            Price,
            Apr,
            FreeShippingThreshold,
            DeliveryTime,
            AgeLimit,
            FoundedYear,
            Rating,
            Spec,
        ] {
            assert_eq!(key.value_hint(), ValueHint::Number, "{key:?}");
        }
        for key in [Deadline, Validity, EventDate] {
            assert_eq!(key.value_hint(), ValueHint::Date, "{key:?}");
        }
        for key in [
            PostalAddress,
            OpeningHours,
            ContactPerson,
            Eligibility,
            CompanyId,
            VatId,
            BankAccount,
            PersonRole,
            Other,
        ] {
            assert_eq!(key.value_hint(), ValueHint::Text, "{key:?}");
        }
    }

    #[test]
    fn priority_sorts_critical_first() {
        let mut priorities = vec![Priority::Low, Priority::Critical, Priority::Medium, Priority::High];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![Priority::Critical, Priority::High, Priority::Medium, Priority::Low]
        );
    }
}
