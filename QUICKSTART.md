# Quick Start Guide

## Installation & Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

Create a `.env` file with your Azure OpenAI credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=Phi-4-multimodal-instruct
```

### 3. Run the Application

```bash
python app.py
```

Visit `http://localhost:7860` in your browser.

## First Query

Try this example query:

```
Research Question: "What are the latest advances in multi-agent reinforcement learning?"
Category: cs.AI - Artificial Intelligence
Number of Papers: 3
```

Click "Analyze Papers" and wait ~1-2 minutes.

## Expected Output

You should see:

1. **Papers Tab**: Table with 3 retrieved papers
2. **Analysis Tab**: Detailed analysis of each paper
3. **Synthesis Tab**:
   - Executive summary
   - Consensus findings (green highlights)
   - Contradictions (yellow highlights)
   - Research gaps
4. **Citations Tab**: APA-formatted references
5. **Stats Tab**: Processing time and cost (~$0.20-0.40)

## Troubleshooting

### Error: "No module named 'xyz'"
```bash
pip install -r requirements.txt --upgrade
```

### Error: "Azure OpenAI authentication failed"
- Check your `.env` file has correct credentials
- Verify your Azure OpenAI deployment name matches your actual deployment

### Error: "Failed to download paper"
- Some arXiv papers may have download issues
- Try a different query or category

### Error: "ChromaDB error"
```bash
rm -rf data/chroma_db/
# Restart the app
```

## Architecture Overview

```
User Query
    ↓
Retriever Agent (arXiv search + PDF processing)
    ↓
Analyzer Agent (RAG-based analysis per paper)
    ↓
Synthesis Agent (Cross-paper comparison)
    ↓
Citation Agent (Validation + APA formatting)
    ↓
Gradio UI (4 output tabs)
```

## Key Features

- **Temperature=0**: Deterministic outputs
- **RAG Grounding**: All claims backed by source text
- **Semantic Caching**: Repeated queries use cache
- **Cost Tracking**: Real-time cost estimates
- **Error Handling**: Graceful failures with user-friendly messages

## Performance Benchmarks

| Papers | Time | Cost | Chunks |
|--------|------|------|--------|
| 3      | ~90s | $0.25 | ~150   |
| 5      | ~120s| $0.40 | ~250   |
| 10     | ~180s| $0.75 | ~500   |

## Next Steps

1. **Customize Categories**: Edit `ARXIV_CATEGORIES` in `app.py`
2. **Adjust Chunking**: Modify `chunk_size` in `utils/pdf_processor.py`
3. **Change Top-K**: Update `top_k` in `rag/retrieval.py`
4. **Add Logging**: Increase log level in agents for debugging

## Deployment to Hugging Face

```bash
# 1. Create a new Space on huggingface.co
# 2. Upload all files
# 3. Add secrets in Space settings:
#    - AZURE_OPENAI_ENDPOINT
#    - AZURE_OPENAI_API_KEY
#    - AZURE_OPENAI_DEPLOYMENT_NAME
# 4. Space will auto-deploy
```

## Support

For issues: https://github.com/yourusername/Multi-Agent-Research-Paper-Analysis-System/issues
