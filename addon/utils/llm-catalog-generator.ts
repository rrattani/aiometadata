import { createHash } from 'crypto';
import consola from 'consola';
import { LRUCache } from 'lru-cache';
import { resolveCatalogModel } from './ai-model-resolver';
import { cacheWrapStaticCatalog, CATALOG_TTL } from '../lib/getCache';

const generationRateLimit = new LRUCache<string, number[]>({ max: 10000, ttl: 60000 });
const logger = consola.withTag('LLMCatalogGenerator');

const TITLE_GENERATION_CONSTRAINT = `TITLE GENERATION:
Generate a short, engaging catalog_title that perfectly summarizes the user's request. Keep the title under 30 characters. Do not use the words "Catalog" or "List".`;

const OUTPUT_FORMAT_CONSTRAINT = `OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema, with no markdown formatting, prefatory text, or trailing explanations:
{
  "type": "object",
  "properties": {
    "catalog_title": {
      "type": "string",
      "description": "The short, polished display title generated from the prompt."
    },
    "catalog": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": { "type": "string", "description": "The exact movie or series title" },
          "release_date": { "type": "string", "description": "The full release date in YYYY-MM-DD format, for example 2026-10-27" },
          "imdb_id": { "type": "string", "description": "The exact IMDb ID starting with 'tt'" }
        },
        "required": ["title", "release_date", "imdb_id"]
      }
    }
  },
  "required": ["catalog_title", "catalog"]
}
Return a JSON instance conforming to that schema, not the schema itself.`;

const DATABASE_ID_ACCURACY_CONSTRAINT = `DATABASE ID ACCURACY (NO HALLUCINATIONS):
You must provide the exact, real-world database IDs for every selected movie or series. Do not guess, sequentialize, or fabricate these numbers under any circumstances.
- IMDb ID: The unique alphanumeric identifier officially assigned to the movie or series by the Internet Movie Database (imdb.com), which must always begin with "tt" followed by a string of digits. Example: Coco is tt2380307.
- If you are not 100% certain of a title's actual IMDb ID, choose a different movie or series where the ID is a known fact.`;

const MOVIE_CONTENT_CONSTRAINT = `MANDATORY CONTENT FORMAT, RUNTIME & QUALITY:
Exclusively curate standalone, feature-length theatrical movies or standalone TV movies with a runtime of at least 75 minutes. Include each movie's correct release year in its release_date. Strictly exclude multi-episode television series, episodic anime, ongoing cartoon shows, short films, and short-form holiday TV specials (such as 25-minute holiday shorts).`;

const SERIES_CONTENT_CONSTRAINT = `MANDATORY CONTENT FORMAT & QUALITY:
Exclusively curate genuine multi-episode television series, including completed and ongoing series and limited series with multiple full-length episodes. Include each series' original first-air date in release_date. Strictly exclude theatrical films, TV movies, one-off specials, pilots that never became a series, short films, and short-form holiday specials.`;

const MAX_CATALOG_ITEMS = 50;
const DEFAULT_CATALOG_TEMPERATURE = 2.0;

const SORTING_CONSTRAINTS: Record<string, string> = {
  'popularity.desc': 'Sort strictly from highest to lowest global popularity and audience recognition. Put the most celebrated, iconic, and widely watched matching titles first.',
  'popularity.asc': 'Sort strictly from lowest to highest global popularity among relevant, well-regarded matching titles. Keep every selection relevant and legitimate; do not pad with low-quality or obscure filler.',
  'primary_release_date.desc': 'Sort strictly by theatrical/TV-movie release date, newest to oldest.',
  'primary_release_date.asc': 'Sort strictly by theatrical/TV-movie release date, oldest to newest.',
  'first_air_date.desc': 'Sort strictly by original first-air date, newest to oldest.',
  'first_air_date.asc': 'Sort strictly by original first-air date, oldest to newest.',
  'title.asc': "Sort strictly by the movie's title in alphabetical order, A to Z.",
  'title.desc': "Sort strictly by the movie's title in reverse alphabetical order, Z to A.",
  'name.asc': 'Sort strictly by the series title in alphabetical order, A to Z.',
  'name.desc': 'Sort strictly by the series title in reverse alphabetical order, Z to A.',
};

interface GenerateCatalogOptions {
  query: string;
  provider: 'gemini' | 'openrouter';
  requestedModel?: string;
  catalogType: 'movie' | 'series';
  catalogId?: string;
  sortBy?: string;
  temperature?: number;
  apiKeys: { gemini?: string; openrouter?: string; tmdb?: string };
  rateLimitKey: string;
  isPreview?: boolean;
}

interface GeneratedCatalogItem {
  title: string;
  release_date: string;
  imdb_id: string;
}

interface ParsedLlmCatalog {
  catalog_title: string;
  catalog: GeneratedCatalogItem[];
}

interface TmdbCatalogResult {
  id: number;
  title: string;
  poster_path: string | null;
  vote_average: number;
  release_date: string;
}

interface GeneratedCatalog {
  catalog_title: string;
  results: TmdbCatalogResult[];
}

function createGenerationError(message: string, statusCode: number): Error & { statusCode: number } {
  return Object.assign(new Error(message), { statusCode });
}

function parseGeneratedCatalog(rawText: string): ParsedLlmCatalog {
  const jsonText = rawText.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  let response: any;
  try {
    response = JSON.parse(jsonText);
  } catch {
    throw createGenerationError('The AI provider returned invalid JSON. Please try again.', 422);
  }

  const title = response?.catalog_title;
  const catalog = response?.catalog;
  const validTitle = typeof title === 'string'
    && title.trim().length > 0
    && title.trim().length < 30
    && !/\b(catalog|list)\b/i.test(title);
  const validCatalog = Array.isArray(catalog)
    && catalog.every((item: any) =>
      item
      && typeof item.title === 'string'
      && item.title.trim().length > 0
      && typeof item.release_date === 'string'
      && /^\d{4}-\d{2}-\d{2}$/.test(item.release_date)
      && !Number.isNaN(Date.parse(`${item.release_date}T00:00:00Z`))
      && new Date(`${item.release_date}T00:00:00Z`).toISOString().slice(0, 10) === item.release_date
      && typeof item.imdb_id === 'string'
      && /^tt\d+$/.test(item.imdb_id)
    );

  if (!validTitle || !validCatalog) {
    throw createGenerationError('The AI provider returned data that did not match the required catalog format. Please try again.', 422);
  }

  return {
    catalog_title: title.trim(),
    catalog: catalog.slice(0, MAX_CATALOG_ITEMS).map((item: GeneratedCatalogItem) => ({
      title: item.title.trim(),
      release_date: item.release_date,
      imdb_id: item.imdb_id,
    })),
  };
}

function normalizeTitle(title: unknown): string {
  return typeof title === 'string'
    ? title.toLocaleLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}]/gu, '')
    : '';
}

function getTitleYear(item: any, catalogType: 'movie' | 'series'): number | null {
  const date = catalogType === 'movie' ? item?.release_date : item?.first_air_date;
  const year = typeof date === 'string' ? Number(date.slice(0, 4)) : NaN;
  return Number.isInteger(year) ? year : null;
}

function matchesLlmItem(
  tmdbItem: any,
  requestedItem: GeneratedCatalogItem,
  catalogType: 'movie' | 'series'
): boolean {
  const requestedTitle = normalizeTitle(requestedItem.title);
  const tmdbTitles = [tmdbItem?.title, tmdbItem?.original_title, tmdbItem?.name, tmdbItem?.original_name]
    .map(normalizeTitle)
    .filter(Boolean);
  return tmdbTitles.includes(requestedTitle)
    && getTitleYear(tmdbItem, catalogType) === Number(requestedItem.release_date.slice(0, 4));
}

async function generateLlmCatalog({
  query,
  provider,
  requestedModel,
  catalogType,
  catalogId,
  sortBy = 'popularity.desc',
  temperature = DEFAULT_CATALOG_TEMPERATURE,
  apiKeys,
  rateLimitKey,
  isPreview = false,
}: GenerateCatalogOptions): Promise<GeneratedCatalog> {
  const isSupportedSort = catalogType === 'movie'
    ? ['popularity.desc', 'popularity.asc', 'primary_release_date.desc', 'primary_release_date.asc', 'title.asc', 'title.desc'].includes(sortBy)
    : ['popularity.desc', 'popularity.asc', 'first_air_date.desc', 'first_air_date.asc', 'name.asc', 'name.desc'].includes(sortBy);
  if (!isSupportedSort) {
    throw createGenerationError(`Unsupported ${catalogType} catalog sort option`, 400);
  }

  const apiKey = provider === 'openrouter' ? apiKeys.openrouter : apiKeys.gemini;
  if (typeof apiKey !== 'string' || !apiKey.trim()) {
    throw createGenerationError(`A ${provider} API key is required`, 400);
  }
  const tmdbApiKey = apiKeys.tmdb;
  if (typeof tmdbApiKey !== 'string' || !tmdbApiKey.trim()) {
    throw createGenerationError('A TMDB API key is required to generate AI catalogs', 400);
  }

  const selectedModel = resolveCatalogModel({ provider, requestedModel });
  const promptDate = new Date().toISOString().slice(0, 10);
  const requestHash = createHash('sha256').update(query.trim()).digest('hex');
  const credentialsHash = createHash('sha256')
    .update([apiKey, tmdbApiKey.trim()].join('\0'))
    .digest('hex');
  let cacheTtl: number | undefined;
  if (!isPreview) {
    const { loadConfigFromDatabase } = require('../lib/configApi');
    const config = await loadConfigFromDatabase(rateLimitKey);
    const catalogConfig = config?.catalogs?.find((catalog: any) =>
      catalogId && catalog.id === catalogId && catalog.type === catalogType
    );
    const customTtl = catalogConfig?.cacheTTL;
    cacheTtl = Number.isFinite(customTtl) && customTtl >= 0
      ? customTtl
      : CATALOG_TTL();
  }
  const cacheKey = `llm.discover:${rateLimitKey}:${catalogId || catalogType}:${provider}:${selectedModel}:${sortBy}:${temperature ?? 'default'}:${cacheTtl ?? 'preview'}:${credentialsHash}:${requestHash}`;

  const mediaType = catalogType === 'movie'
    ? 'movie'
    : 'TV series';
  const contentConstraint = catalogType === 'movie'
    ? MOVIE_CONTENT_CONSTRAINT
    : SERIES_CONTENT_CONSTRAINT;
  const prompt = `Today's date is ${promptDate}.
You are an expert media curator. Your task is to build a highly curated ${mediaType} catalog based on the user's request.

User's request: ${query.trim()}

HARD CONSTRAINTS:
${DATABASE_ID_ACCURACY_CONSTRAINT}

${contentConstraint}

OUTPUT HANDLING & SORTING:
Generate exactly ${MAX_CATALOG_ITEMS} distinct, highly relevant titles when that many eligible titles with verifiable IMDb IDs are available. Return fewer only when needed to preserve the content and ID-accuracy requirements; never exceed ${MAX_CATALOG_ITEMS} items. ${SORTING_CONSTRAINTS[sortBy]}
Preserve the requested order exactly in the output. Do not add titles that violate the content requirements to reach the item limit.

${TITLE_GENERATION_CONSTRAINT}

${OUTPUT_FORMAT_CONSTRAINT}`;

  const generate = async (): Promise<GeneratedCatalog> => {
    const now = Date.now();
    const recentRequests = (generationRateLimit.get(rateLimitKey) || []).filter(timestamp => now - timestamp < 60000);
    if (recentRequests.length >= 5) {
      throw createGenerationError('Too many requests. Please wait a moment before trying again.', 429);
    }
    recentRequests.push(now);
    generationRateLimit.set(rateLimitKey, recentRequests);

    const client = provider === 'openrouter'
      ? require('./openrouter-client')
      : require('./gemini-client');
    const result = await client.generateContent({
      apiKey: apiKey.trim(),
      model: selectedModel,
      prompt,
      temperature,
      timeout: 90000,
    });

    if (typeof result.text !== 'string' || !result.text.trim()) {
      throw createGenerationError('The AI provider returned an empty response. Please try again.', 502);
    }

    const catalog = parseGeneratedCatalog(result.text);
    const moviedb = require('../lib/getTmdb');
    const tmdbConfig = { apiKeys: { tmdb: tmdbApiKey.trim() } };
    const tmdbResults = await Promise.all(catalog.catalog.map(async item => {
      try {
        const found = await moviedb.find(
          { id: item.imdb_id, external_source: 'imdb_id' },
          tmdbConfig
        );
        const findResults = catalogType === 'movie' ? found?.movie_results : found?.tv_results;
        const findMatch = Array.isArray(findResults)
          ? findResults.find((entry: any) => matchesLlmItem(entry, item, catalogType))
          : null;

        let tmdbId = findMatch?.id;
        if (!tmdbId) {
          const releaseYear = Number(item.release_date.slice(0, 4));
          const searchParams = catalogType === 'movie'
            ? { query: item.title, year: releaseYear, primary_release_year: releaseYear }
            : { query: item.title, year: releaseYear, first_air_date_year: releaseYear };
          const search = catalogType === 'movie'
            ? await moviedb.searchMovie(searchParams, tmdbConfig)
            : await moviedb.searchTv(searchParams, tmdbConfig);
          const searchResults = Array.isArray(search?.results) ? search.results : [];
          const searchMatch = searchResults.find((entry: any) => matchesLlmItem(entry, item, catalogType));
          tmdbId = searchMatch?.id;
        }

        if (!Number.isInteger(tmdbId)) {
          logger.warn(`TMDB could not verify title/year for IMDb ID ${item.imdb_id}: ${item.title} (${item.release_date.slice(0, 4)})`);
          return null;
        }

        const details = catalogType === 'movie'
          ? await moviedb.movieInfo({ id: tmdbId }, tmdbConfig)
          : await moviedb.tvInfo({ id: tmdbId }, tmdbConfig);
        if (!details?.id || !matchesLlmItem(details, item, catalogType)) {
          logger.warn(`TMDB details did not match ${item.title} (${item.release_date.slice(0, 4)}) for IMDb ID ${item.imdb_id}`);
          return null;
        }

        return {
          id: details.id,
          title: catalogType === 'movie'
            ? details.title || details.original_title
            : details.name || details.original_name,
          poster_path: details.poster_path || null,
          vote_average: details.vote_average || 0,
          release_date: catalogType === 'movie' ? details.release_date : details.first_air_date,
        } satisfies TmdbCatalogResult;
      } catch (error: any) {
        logger.warn(`TMDB lookup failed for ${item.imdb_id}: ${error.message}`);
        return null;
      }
    }));

    return {
      catalog_title: catalog.catalog_title,
      results: tmdbResults.filter((item): item is TmdbCatalogResult => item !== null),
    };
  };

  if (isPreview) return generate();
  const cachedCatalog = await cacheWrapStaticCatalog(rateLimitKey, cacheKey, generate, {
    resultClassifier: (result: GeneratedCatalog) => Array.isArray(result?.results) && result.results.length > 0
      ? { type: 'SUCCESS', ttl: null }
      : { type: 'SKIP_CACHE', ttl: 0 },
  }, cacheTtl ?? CATALOG_TTL());
  if (!cachedCatalog || !Array.isArray(cachedCatalog.results)) {
    throw createGenerationError('Unable to load the generated AI catalog from cache.', 500);
  }
  return cachedCatalog;
}

export { generateLlmCatalog };
module.exports = { generateLlmCatalog };
