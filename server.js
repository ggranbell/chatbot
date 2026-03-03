const ollamaClass = require('ollama');
const ollama = new ollamaClass.Ollama();
const expressRateLimit = require('express-rate-limit');
const lancedb = require("@lancedb/lancedb");
const path = require('path'); 
const fs = require('fs');
const os = require('os');
const { execFile } = require('child_process');
const { promisify } = require('util');
const multer = require('multer');




const dotenv = require('dotenv');
dotenv.config();

const express = require('express');
const app = express();
const port = 8080 || process.env.PORT;
const validAuthKey = process.env.AUTH_KEY;
const VECTOR_TOP_K = Number(process.env.VECTOR_TOP_K || 24);
const FINAL_CONTEXT_K = 5;
const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL || 'ministral-3:3b';
const CHAT_MODEL_SHORT = process.env.OLLAMA_CHAT_MODEL_SHORT || CHAT_MODEL;
const CHAT_MODEL_LONG = process.env.OLLAMA_CHAT_MODEL_LONG || CHAT_MODEL;
const CHAT_MODEL_LONG_THRESHOLD = Number(process.env.OLLAMA_CHAT_MODEL_LONG_THRESHOLD || 1800);
const CHAT_MODEL_FALLBACKS = (process.env.OLLAMA_CHAT_MODEL_FALLBACKS || '')
    .split(',')
    .map(model => model.trim())
    .filter(Boolean);
const EMBEDDING_MODEL = process.env.OLLAMA_EMBED_MODEL || 'nomic-embed-text';
const EMBEDDING_MODEL_FALLBACKS = (process.env.OLLAMA_EMBED_MODEL_FALLBACKS || '')
    .split(',')
    .map(model => model.trim())
    .filter(Boolean);
const MAX_CHUNKS_PER_LABEL = Number(process.env.MAX_CHUNKS_PER_LABEL || 2);
const MIN_RELATIVE_HYBRID_SCORE = Number(process.env.MIN_RELATIVE_HYBRID_SCORE || 0.45);
const rateLimit = expressRateLimit({
    windowMs: 15 * 60 * 1000, 
    max: 10, 
    message: "Too many requests, please try again later."
})


app.use(express.json());
const upload = multer({ storage: multer.memoryStorage() });
app.use(express.static(path.join(__dirname, 'src')));
const execFileAsync = promisify(execFile);
const venvPython = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
const pythonExecutable = process.env.PYTHON_EXECUTABLE || (fs.existsSync(venvPython) ? venvPython : 'python');
const tesseractCmd = process.env.TESSERACT_CMD;
const modelInventoryCache = {
    expiresAt: 0,
    names: null,
};

function estimateTokenCount(text) {
    if (!text || typeof text !== 'string') return 0;
    return Math.ceil(text.length / 4);
}

function uniqueOrdered(values) {
    return [...new Set(values.filter(Boolean))];
}

function matchInstalledModel(candidateModel, installedNames) {
    if (!candidateModel || !installedNames?.size) return false;
    if (installedNames.has(candidateModel)) return true;
    if (candidateModel.includes(':')) return false;
    for (const installed of installedNames) {
        if (installed === candidateModel || installed.startsWith(`${candidateModel}:`)) {
            return true;
        }
    }
    return false;
}

async function getInstalledModelNames() {
    const now = Date.now();
    if (modelInventoryCache.names && modelInventoryCache.expiresAt > now) {
        return modelInventoryCache.names;
    }

    try {
        const listResponse = await ollama.list();
        const names = new Set(
            (listResponse?.models || [])
                .map(model => model?.name || model?.model)
                .filter(Boolean)
        );
        modelInventoryCache.names = names;
        modelInventoryCache.expiresAt = now + 30_000;
        return names;
    } catch {
        return null;
    }
}

async function getAdaptiveChatModelCandidates(promptText = '') {
    const estimatedTokens = estimateTokenCount(promptText);
    const primaryModel = estimatedTokens >= CHAT_MODEL_LONG_THRESHOLD ? CHAT_MODEL_LONG : CHAT_MODEL_SHORT;
    const configuredCandidates = uniqueOrdered([primaryModel, CHAT_MODEL, ...CHAT_MODEL_FALLBACKS]);
    const installedNames = await getInstalledModelNames();
    if (!installedNames?.size) return configuredCandidates;

    const installedCandidates = configuredCandidates.filter(model => matchInstalledModel(model, installedNames));
    return installedCandidates.length ? installedCandidates : configuredCandidates;
}

async function getAdaptiveEmbeddingModelCandidates() {
    const configuredCandidates = uniqueOrdered([EMBEDDING_MODEL, ...EMBEDDING_MODEL_FALLBACKS]);
    const installedNames = await getInstalledModelNames();
    if (!installedNames?.size) return configuredCandidates;

    const installedCandidates = configuredCandidates.filter(model => matchInstalledModel(model, installedNames));
    return installedCandidates.length ? installedCandidates : configuredCandidates;
}

function extractQueryTerms(text) {
    if (!text || typeof text !== 'string') return new Set();
    const stopwords = new Set([
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'to', 'of', 'in', 'on', 'at', 'by',
        'from', 'with', 'without', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that',
        'these', 'those', 'as', 'into', 'about', 'than', 'so', 'such', 'not', 'no', 'yes', 'can', 'could', 'should',
        'would', 'will', 'may', 'might', 'must', 'do', 'does', 'did', 'done', 'have', 'has', 'had', 'having', 'you',
        'your', 'yours', 'we', 'our', 'ours', 'they', 'their', 'theirs', 'he', 'him', 'his', 'she', 'her', 'hers',
        'them', 'who', 'whom', 'which', 'what', 'when', 'where', 'why', 'how', 'all', 'any', 'every', 'more', 'most'
    ]);

    const tokens = (text.toLowerCase().match(/[a-z][a-z0-9_-]{2,}/g) || [])
        .filter(token => !stopwords.has(token));
    return new Set(tokens);
}

function getLexicalOverlapRatio(queryTerms, candidateText) {
    if (!queryTerms.size || !candidateText) return 0;
    const candidateTokens = new Set((candidateText.toLowerCase().match(/[a-z][a-z0-9_-]{2,}/g) || []));
    if (!candidateTokens.size) return 0;

    let overlapCount = 0;
    queryTerms.forEach(term => {
        if (candidateTokens.has(term)) overlapCount += 1;
    });

    return overlapCount / Math.max(queryTerms.size, 1);
}

function getVectorSimilarity(distanceValue) {
    if (typeof distanceValue !== 'number' || Number.isNaN(distanceValue)) return 0;
    return 1 / (1 + Math.max(distanceValue, 0));
}

function normalizeJsonValue(value) {
    if (typeof value === 'bigint') {
        return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString();
    }

    if (Array.isArray(value)) {
        return value.map(normalizeJsonValue);
    }

    if (value && typeof value === 'object') {
        return Object.fromEntries(
            Object.entries(value).map(([key, objectValue]) => [key, normalizeJsonValue(objectValue)])
        );
    }

    return value;
}

async function getTableColumnNames(table) {
    try {
        const schema = typeof table?.schema === 'function' ? await table.schema() : table?.schema;
        const fields = schema?.fields || [];
        return new Set(fields.map(field => field.name));
    } catch {
        return new Set();
    }
}

function getRowKeywords(row) {
    if (row?.keywords) return row.keywords;
    if (row?.metadata?.keywords) return row.metadata.keywords;
    return 'N/A';
}

function isModelNotFoundError(error) {
    return error?.status_code === 404 && typeof error?.error === 'string' && /model\s+'.+'\s+not found/i.test(error.error);
}

function extractEmbeddingFromResponse(embedResponse) {
    const vector = embedResponse?.embedding
        || (Array.isArray(embedResponse?.embeddings) ? embedResponse.embeddings[0] : null);
    return Array.isArray(vector) && vector.length ? vector : null;
}

async function ollamaChatWithFallback(payload, promptText = '') {
    const candidateModels = await getAdaptiveChatModelCandidates(promptText);
    let lastError;

    for (const model of candidateModels) {
        try {
            const response = await ollama.chat({ ...payload, model });
            return { response, modelUsed: model };
        } catch (error) {
            lastError = error;
            if (!isModelNotFoundError(error)) {
                throw error;
            }
        }
    }

    throw lastError;
}

async function ollamaEmbeddingWithFallback(prompt) {
    const candidateModels = await getAdaptiveEmbeddingModelCandidates();
    let lastError;

    for (const model of candidateModels) {
        try {
            const response = await ollama.embeddings({ model, prompt });
            const embedding = extractEmbeddingFromResponse(response);
            if (!embedding) {
                throw new Error(`Embedding response is empty for model '${model}'.`);
            }
            return { embedding, modelUsed: model };
        } catch (error) {
            lastError = error;
            if (!isModelNotFoundError(error)) {
                throw error;
            }
        }
    }

    throw lastError;
}

app.get('/', (req, res) => {
    res.send('Server is online.')
})

app.post('/upload-knowledge-base', upload.single('file'), async(req, res) =>{
    let tempFilePath = null;
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        if (req.file.mimetype !== 'application/pdf') {
            return res.status(400).json({ error: "Only PDF files are allowed" });
        }

        const tempDir = path.join(os.tmpdir(), 'mola-chatbot-ingest');
        fs.mkdirSync(tempDir, { recursive: true });

        const safeFileName = `${Date.now()}-${req.file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
        tempFilePath = path.join(tempDir, safeFileName);
        fs.writeFileSync(tempFilePath, req.file.buffer);

        const scriptPath = path.join(__dirname, 'pdf_to_vectordb.py');
        const label = req.body?.label;

        const commandArgs = [
            scriptPath,
            '--pdf-path', tempFilePath,
            '--db-path', 'data/vector-db',
            '--table', 'knowledge_base',
            '--embedding-model', 'nomic-embed-text',
        ];

        if (label) {
            commandArgs.push('--label', label);
        }

        if (tesseractCmd) {
            commandArgs.push('--tesseract-cmd', tesseractCmd);
        }

        const { stdout, stderr } = await execFileAsync(pythonExecutable, commandArgs, {
            cwd: __dirname,
            timeout: 10 * 60 * 1000,
            maxBuffer: 1024 * 1024,
        });

        const output = `${stdout || ''}\n${stderr || ''}`.trim();
        const match = output.match(/Added\s+(\d+)\s+rows/i);
        const rowsAdded = match ? Number(match[1]) : null;

        res.status(200).json({ 
            success: true, 
            message : rowsAdded !== null
                ? `Knowledge base updated. Added ${rowsAdded} chunks.`
                : 'Knowledge base updated from uploaded PDF.',
            details: output || undefined,
        });
    } catch (err) {
        console.log(err.message)
        const details = err?.stderr || err?.stdout || err.message;
        res.status(500).json({ error: "Failed to ingest PDF into vector DB", details });
    } finally {
        if (tempFilePath && fs.existsSync(tempFilePath)) {
            fs.unlinkSync(tempFilePath);
        }
    }
})

app.get('/admin/db-rows', async (req, res) => {
    const authKey = req.headers['x-auth-key'];
    if (authKey !== validAuthKey) {
        return res.status(401).json({ message: 'Authentication key is invalid.' });
    }

    try {
        const db = await lancedb.connect("data/vector-db");
        const tableNames = new Set(await db.tableNames());

        if (!tableNames.has('knowledge_base')) {
            return res.status(200).json({ success: true, total: 0, rows: [] });
        }

        const table = await db.openTable("knowledge_base");
        const tableColumns = await getTableColumnNames(table);
        const limitParam = Number(req.query.limit);
        const limit = Number.isFinite(limitParam) ? Math.min(Math.max(limitParam, 1), 100) : 20;

        const baseRequestedColumns = ['id', 'label', 'keywords', 'text', 'metadata'];
        const selectedColumns = baseRequestedColumns.filter(column => tableColumns.has(column));

        if (!selectedColumns.length) {
            return res.status(200).json({ success: true, total: 0, rows: [] });
        }

        const rows = await table
            .query()
            .select(selectedColumns)
            .limit(limit)
            .toArray();

        let total = rows.length;
        if (typeof table.countRows === 'function') {
            const counted = await table.countRows();
            total = normalizeJsonValue(counted);
        }

        const normalizedRows = rows.map(row => {
            const safeRow = {
                id: row.id ?? null,
                label: row.label ?? 'N/A',
                keywords: getRowKeywords(row),
                text: row.text ?? '',
            };
            return normalizeJsonValue(safeRow);
        });

        return res.status(200).json({
            success: true,
            total,
            rows: normalizedRows,
        });
    } catch (error) {
        console.error("DB rows error:", error);
        return res.status(500).json({ message: 'Failed to read database rows.' });
    }
});

app.post('/admin/db-truncate', async (req, res) => {
    const authKey = req.headers['x-auth-key'];
    if (authKey !== validAuthKey) {
        return res.status(401).json({ message: 'Authentication key is invalid.' });
    }

    try {
        const db = await lancedb.connect("data/vector-db");
        const tableNames = new Set(await db.tableNames());

        if (tableNames.has('knowledge_base')) {
            await db.dropTable('knowledge_base');
        }

        return res.status(200).json({
            success: true,
            message: 'knowledge_base table truncated.',
        });
    } catch (error) {
        console.error("DB truncate error:", error);
        return res.status(500).json({ message: 'Failed to truncate database table.' });
    }
});

app.post('/ask-ai', rateLimit, async (req, res) => {
    const authKey = req.headers['x-auth-key'];
    // console.log(authKey)
    const { message } = req.body;
    const isAuthKeyValid = authKey === validAuthKey;
    const payload = {
        messages : [{
            role : 'user',
            content : message
        }]
    }

    // console.log(`Client's Key: ${authKey} \nValid Key: ${validAuthKey}`)
    if(!isAuthKeyValid) {
        return res.status(401).json({
            message : 'Authentication key is invalid.'
        });
    }

    const { response, modelUsed } = await ollamaChatWithFallback(payload, message || '');

    return res.status(200).json({
        message : response.message.content,
        model: modelUsed,
    });

})

app.post('/ask-ai-with-vector', async (req, res) => {
    const authKey = req.headers['x-auth-key'];
    const { message, context } = req.body;
    const requestStart = Date.now();
    // console.log('here')

    if (authKey !== validAuthKey) {
        return res.status(401).json({ message: 'Authentication key is invalid.' });
    }

    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ message: 'Message is required.' });
    }

    try {
        const timeLogs = {};

        const dbStart = Date.now();
        const db = await lancedb.connect("data/vector-db");
        const tableNames = new Set(await db.tableNames());
        timeLogs.dbConnectMs = Date.now() - dbStart;

        if (!tableNames.has('knowledge_base')) {
            return res.status(404).json({ message: 'Knowledge base is empty. Please upload a PDF first.' });
        }

        const table = await db.openTable("knowledge_base");

        const historyContext = Array.isArray(context)
            ? context.filter(item => typeof item === 'string' && item.trim()).slice(-3).join('\n')
            : '';

        const retrievalQuery = historyContext
            ? `Conversation context:\n${historyContext}\n\nCurrent user question:\n${message}`
            : message;


        const embeddingStart = Date.now();
        const { embedding, modelUsed: embeddingModelUsed } = await ollamaEmbeddingWithFallback(retrievalQuery);
        timeLogs.embeddingMs = Date.now() - embeddingStart;
        timeLogs.embeddingModel = embeddingModelUsed;

        const tableColumns = await getTableColumnNames(table);
        const retrievalColumns = ['text', 'keywords', 'label', 'metadata'].filter(column => tableColumns.has(column));

        const vectorSearchStart = Date.now();
        const searchResults = await table
            .search(embedding)
            .limit(VECTOR_TOP_K)
            .select(retrievalColumns)
            .toArray();
        timeLogs.vectorSearchMs = Date.now() - vectorSearchStart;

        const queryTerms = extractQueryTerms(message);
        const distances = searchResults
            .map(item => item._distance)
            .filter(distance => typeof distance === 'number' && Number.isFinite(distance));
        const minDistance = distances.length ? Math.min(...distances) : null;
        const maxDistance = distances.length ? Math.max(...distances) : null;

        const rerankStart = Date.now();
        const rerankedCandidates = searchResults
            .map(item => {
                const keywordText = getRowKeywords(item);
                const candidateText = `${item.text || ''} ${keywordText || ''} ${item.label || ''}`.trim();
                const lexicalScore = getLexicalOverlapRatio(queryTerms, candidateText);
                const keywordCoverageScore = getLexicalOverlapRatio(queryTerms, `${keywordText || ''} ${item.label || ''}`.trim());

                let normalizedVectorScore = getVectorSimilarity(item._distance);
                if (typeof item._distance === 'number' && Number.isFinite(item._distance) && minDistance !== null && maxDistance !== null) {
                    if (maxDistance > minDistance) {
                        const minMaxNormalized = (maxDistance - item._distance) / (maxDistance - minDistance);
                        normalizedVectorScore = (normalizedVectorScore * 0.6) + (minMaxNormalized * 0.4);
                    }
                }

                const hybridScore = (normalizedVectorScore * 0.62) + (lexicalScore * 0.25) + (keywordCoverageScore * 0.13);

                return {
                    ...item,
                    _hybridScore: hybridScore,
                    _vectorScore: normalizedVectorScore,
                    _lexicalScore: lexicalScore,
                    _keywordCoverageScore: keywordCoverageScore,
                };
            })
            .sort((a, b) => b._hybridScore - a._hybridScore);

        const bestHybridScore = rerankedCandidates[0]?._hybridScore ?? 0;
        const minAllowedScore = bestHybridScore > 0 ? bestHybridScore * MIN_RELATIVE_HYBRID_SCORE : 0;
        const filteredCandidates = rerankedCandidates.filter(item => item._hybridScore >= minAllowedScore);

        const selectedByDiversity = [];
        const labelCounts = new Map();
        for (const candidate of filteredCandidates) {
            const label = (candidate.label || 'N/A').toLowerCase();
            const currentCount = labelCounts.get(label) || 0;
            if (currentCount >= MAX_CHUNKS_PER_LABEL) {
                continue;
            }
            selectedByDiversity.push(candidate);
            labelCounts.set(label, currentCount + 1);
            if (selectedByDiversity.length >= FINAL_CONTEXT_K) {
                break;
            }
        }

        const fallbackPool = filteredCandidates.length ? filteredCandidates : rerankedCandidates;
        const rankedResults = selectedByDiversity.length
            ? selectedByDiversity
            : fallbackPool.slice(0, FINAL_CONTEXT_K);

        if (rankedResults.length < FINAL_CONTEXT_K) {
            for (const candidate of fallbackPool) {
                if (rankedResults.includes(candidate)) continue;
                rankedResults.push(candidate);
                if (rankedResults.length >= FINAL_CONTEXT_K) break;
            }
        }

        timeLogs.rerankMs = Date.now() - rerankStart;
        timeLogs.vectorCandidates = searchResults.length;
        timeLogs.rerankedCandidates = rerankedCandidates.length;

        if (!rankedResults.length) {
            return res.status(404).json({ message: 'No relevant knowledge base context found.' });
        }

        const contextText = rankedResults
            .map(r => `[Label: ${r.label || 'N/A'}]\n[Keywords: ${getRowKeywords(r)}]\n[Text: ${r.text}]`)
            .join("\n---\n");

        const selectedChunks = rankedResults.map((chunk, index) => ({
            rank: index + 1,
            label: chunk.label || 'N/A',
            keywords: getRowKeywords(chunk),
            hybridScore: Number((chunk._hybridScore || 0).toFixed(6)),
            vectorScore: Number((chunk._vectorScore || 0).toFixed(6)),
            lexicalScore: Number((chunk._lexicalScore || 0).toFixed(6)),
            keywordCoverageScore: Number((chunk._keywordCoverageScore || 0).toFixed(6)),
            vectorDistance: typeof chunk._distance === 'number' ? Number(chunk._distance.toFixed(6)) : null,
            text: chunk.text || '',
        }));

        const prompt = `
            You are a helpful assistant. Answer the question using ONLY the context provided below.

            CONTEXT:
            ${contextText}

            QUESTION:
            ${message}
            
            Strict Operational Rules
            Zero Inference: Do not use any prior knowledge, external facts, or logical assumptions not explicitly stated in the text.
            Ground Truth Only: If the answer is not contained within the CONTEXT, you must state: 'I am sorry, but the provided context does not contain enough information to answer this question.'
            No Hallucinations: Do not invent names, dates, or details to fill gaps in the narrative.
            Literal Accuracy: Prioritize direct quotes or paraphrasing that maintains the original meaning without adding outside information.
            
            FORMATTING INSTRUCTIONS:
            - Your response must be written entirely in Markdown format.
            - *Extreme Breathability*: Ensure there is a full empty line (double line break) between EVERY single sentence or bullet point.
            - Use headers (##) to separate different topics.
            - Use bold text for key terms to make them stand out.
            - Keep paragraphs extremely short (1-2 sentences max).
            - Provide ONLY the Markdown content; no introductory or closing filler.

        `;


        const generationStart = Date.now();
        const { response, modelUsed } = await ollamaChatWithFallback({
            messages: [{ role: 'user', content: prompt }],
            stream: false, // Ensure streaming is off
        }, prompt);
        timeLogs.generationMs = Date.now() - generationStart;
        timeLogs.totalMs = Date.now() - requestStart;

        // Send as a JSON object
        res.json({
            success: true,
            model: modelUsed,
            response : response.message.content,
            retrievalCount: rankedResults.length,
            selectedChunks,
            timeLogs,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error("Endpoint Error:", error);
        if (!res.headersSent) {
            const details = typeof error?.message === 'string' ? error.message : undefined;
            res.status(500).json({ message: 'Internal Server Error', details });
        }
        else res.end();
    }
});

app.listen(port, () => {
    console.log(`Server is online on port: ${port}`)
})