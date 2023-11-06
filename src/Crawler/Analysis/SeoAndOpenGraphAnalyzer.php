<?php

/*
 * This file is part of the SiteOne Website Crawler.
 *
 * (c) Ján Regeš <jan.reges@siteone.cz>
 */

declare(strict_types=1);

namespace Crawler\Analysis;

use Crawler\Analysis\Result\HeadingTreeItem;
use Crawler\Analysis\Result\SeoAndOpenGraphResult;
use Crawler\Components\SuperTable;
use Crawler\Components\SuperTableColumn;
use Crawler\Crawler;
use Crawler\Options\Group;
use Crawler\Options\Option;
use Crawler\Options\Options;
use Crawler\Options\Type;
use Crawler\Result\Status;
use Crawler\Result\VisitedUrl;
use Crawler\Utils;
use DOMDocument;

class SeoAndOpenGraphAnalyzer extends BaseAnalyzer implements Analyzer
{
    const SUPER_TABLE_SEO = 'seo';
    const SUPER_TABLE_OPEN_GRAPH = 'open-graph';
    const SUPER_TABLE_SEO_HEADINGS = 'seo-headings';

    const GROUP_SEO_AND_OPENGRAPH_ANALYZER = 'seo-and-opengraph-analyzer';

    protected int $maxHeadingLevel = 3;

    private bool $hasOgTags = false;
    private bool $hasTwitterTags = false;

    public function shouldBeActivated(): bool
    {
        return true;
    }

    public function analyze(): void
    {
        $htmlUrls = array_filter($this->status->getVisitedUrls(), function ($visitedUrl) {
            return $visitedUrl->statusCode === 200 && !$visitedUrl->isExternal && $visitedUrl->contentType === Crawler::CONTENT_TYPE_ID_HTML;
        });

        $urlResults = $this->getSeoAndOpenGraphResults($htmlUrls);

        // check if there are any OG or Twitter tags
        foreach ($urlResults as $urlResult) {
            if ($this->hasOgTags && $this->hasTwitterTags) {
                break;
            }

            if ($urlResult->ogTitle !== null || $urlResult->ogDescription !== null || $urlResult->ogImage !== null) {
                $this->hasOgTags = true;
            }
            if ($urlResult->twitterCard !== null || $urlResult->twitterTitle !== null || $urlResult->twitterDescription !== null || $urlResult->twitterImage !== null) {
                $this->hasTwitterTags = true;
            }
        }

        $s = microtime(true);
        $this->analyzeSeo($urlResults);
        $this->measureExecTime(__CLASS__, 'analyzeSeo', $s);

        $s = microtime(true);
        $this->analyzeOpenGraph($urlResults);
        $this->measureExecTime(__CLASS__, 'analyzeOpenGraph', $s);

        $s = microtime(true);
        $this->analyzeHeadings($urlResults);

        $this->measureExecTime(__CLASS__, 'analyzeHeadings', $s);
    }

    /**
     * @param VisitedUrl[] $htmlUrls
     * @return SeoAndOpenGraphResult[]
     */
    private function getSeoAndOpenGraphResults(array $htmlUrls): array
    {
        $results = [];
        $robotsTxtContent = Status::getRobotsTxtContent();
        foreach ($htmlUrls as $visitedUrl) {
            $htmlBody = $this->status->getStorage()->load($visitedUrl->uqId);

            $dom = new DOMDocument();
            @$dom->loadHTML(@mb_convert_encoding($htmlBody, 'HTML-ENTITIES', 'UTF-8'));

            $urlPath = parse_url($visitedUrl->url, PHP_URL_PATH);
            $urlQuery = parse_url($visitedUrl->url, PHP_URL_QUERY);
            $urlPathAndQuery = $urlPath . ($urlQuery ? '?' . $urlQuery : '');

            $urlResult = SeoAndOpenGraphResult::getFromHtml($visitedUrl->uqId, $urlPathAndQuery, $dom, $robotsTxtContent, $this->maxHeadingLevel);
            $results[] = $urlResult;
        }
        return $results;
    }

    /**
     * @param SeoAndOpenGraphResult[] $urlResults
     * @return void
     */
    private function analyzeSeo(array $urlResults): void
    {
        $superTable = new SuperTable(
            self::SUPER_TABLE_SEO,
            "SEO metadata",
            "No URLs.",
            [
                new SuperTableColumn('urlPathAndQuery', 'URL', 50, null, null, true),
                new SuperTableColumn('indexing', 'Indexing', 17, null, function (SeoAndOpenGraphResult $urlResult) {
                    if ($urlResult->deniedByRobotsTxt) {
                        return Utils::getColorText('DENY (robots.txt)', 'magenta');
                    } elseif ($urlResult->robotsIndex === SeoAndOpenGraphResult::ROBOTS_NOINDEX) {
                        return Utils::getColorText('DENY (meta)', 'magenta');
                    } else {
                        return 'Allowed';
                    }
                }, false, false),
                new SuperTableColumn('title', 'Title', 30, null, null, true),
                new SuperTableColumn('h1', 'H1', 30, function ($value) {
                    if (!$value) {
                        return Utils::getColorText('Missing H1', 'red', true);
                    }
                    return $value;
                }, null, true, false),
                new SuperTableColumn('description', 'Description', 30, null, null, true),
                new SuperTableColumn('keywords', 'Keywords', 30, null, null, true),
            ], true, 'urlPathAndQuery', 'ASC'
        );

        // set initial URL (required for urlPath column and active link building)
        $superTable->setInitialUrl($this->status->getOptions()->url);

        $superTable->setData($urlResults);
        $this->status->addSuperTableAtBeginning($superTable);
        $this->output->addSuperTable($superTable);
    }

    /**
     * @param SeoAndOpenGraphResult[] $urlResults
     * @return void
     */
    private function analyzeOpenGraph(array $urlResults): void
    {
        $columns = [
            new SuperTableColumn('urlPathAndQuery', 'URL', 50, null, null, true),
        ];

        if ($this->hasOgTags) {
            $columns[] = new SuperTableColumn('ogTitle', 'OG Title', 32, null, null, true);
            $columns[] = new SuperTableColumn('ogDescription', 'OG Description', 32, null, null, true);
            $columns[] = new SuperTableColumn('ogImage', 'OG Image', 18, null, null, true);
        }
        if ($this->hasTwitterTags) {
            $columns[] = new SuperTableColumn('twitterTitle', 'Twitter Title', 32, null, null, true);
            $columns[] = new SuperTableColumn('twitterDescription', 'Twitter Description', 32, null, null, true);
            $columns[] = new SuperTableColumn('twitterImage', 'Twitter Image', 18, null, null, true);
        }

        $superTable = new SuperTable(
            self::SUPER_TABLE_OPEN_GRAPH,
            "OpenGraph metadata",
            "No URLs with OpenGraph data (og:* or twitter:* meta tags).",
            $columns,
            true,
            'urlPathAndQuery',
            'ASC'
        );

        $superTableData = [];
        if ($this->hasOgTags || $this->hasTwitterTags) {
            $superTableData = $urlResults;
        }

        // set initial URL (required for urlPath column and active link building)
        $superTable->setInitialUrl($this->status->getOptions()->url);

        $superTable->setData($superTableData);
        $this->status->addSuperTableAtBeginning($superTable);
        $this->output->addSuperTable($superTable);
    }

    /**
     * @param SeoAndOpenGraphResult[] $urlResults
     * @return void
     */
    private function analyzeHeadings(array $urlResults): void
    {
        $superTable = new SuperTable(
            self::SUPER_TABLE_SEO_HEADINGS,
            "Headings structure",
            "No URLs to analyze heading structure.",
            [
                new SuperTableColumn('headings', 'Headings structure', 80, null, function (SeoAndOpenGraphResult $urlResult, string $renderInfo) {
                    if (!$urlResult->headingTreeItems) {
                        return '';
                    }
                    if ($renderInfo === SuperTable::RENDER_INTO_CONSOLE) {
                        return HeadingTreeItem::getHeadingTreeTxtList($urlResult->headingTreeItems);
                    } else {
                        return HeadingTreeItem::getHeadingTreeUlLiList($urlResult->headingTreeItems);
                    }
                }, true, false),
                new SuperTableColumn('headingsCount', 'Count', 5),
                new SuperTableColumn('headingsErrorsCount', 'Errors', 6, function ($value) {
                    return Utils::getColorText(strval($value), $value > 0 ? 'red' : 'green', true);
                }, null, false, false),
                new SuperTableColumn('urlPathAndQuery', 'URL', 30, null, null, true),
            ], true, 'urlPathAndQuery', 'ASC'
        );

        // set initial URL (required for urlPath column and active link building)
        $superTable->setInitialUrl($this->status->getOptions()->url);

        $superTable->setData($urlResults);
        $this->status->addSuperTableAtBeginning($superTable);
        $this->output->addSuperTable($superTable);
    }

    public static function getOptions(): Options
    {
        $options = new Options();
        $options->addGroup(new Group(
            self::GROUP_SEO_AND_OPENGRAPH_ANALYZER,
            'SEO and OpenGraph analyzer', [
            new Option('--max-heading-level', null, 'maxHeadingLevel', Type::INT, false, 'Maximal analyzer heading level from 1 to 6.', 3, false, false, [1, 6]),
        ]));
        return $options;
    }

    public function getOrder(): int
    {
        return 113;
    }

}