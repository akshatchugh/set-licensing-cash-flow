# AI Integration with Azure OpenAI

## Overview

The contract extraction functionality has been upgraded to use Azure OpenAI with Langchain for intelligent contract data extraction. The system automatically falls back to pattern matching if AI is unavailable.

## Configuration

The system uses environment variables from `.env` file:
- `AZURE_ENDPOINT` - Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY` - API key for Azure OpenAI
- `AZURE_DEPLOYMENT` - Deployment name (default: 'gpt-4')
- `AZURE_OPENAI_API_VERSION` - API version (default: '2024-02-15-preview')

## How It Works

1. **AI Extraction (Primary)**: When a contract PDF is uploaded:
   - Text is extracted from the PDF using `pdfplumber`
   - The text is sent to Azure OpenAI via Langchain
   - AI extracts structured contract data (Client Name, Deal Value, Dates, Payment Terms, etc.)
   - Returns JSON with extracted fields

2. **Fallback (Secondary)**: If AI extraction fails or is unavailable:
   - Falls back to pattern matching extraction
   - Uses regex patterns to find contract information
   - Provides basic extraction capabilities

## Extracted Fields

The AI extracts the following contract fields:
- **Client Name**: Name of the client/party
- **Deal Value**: Total contract value (converts lakhs/crores to numeric)
- **Currency**: Currency type (INR, USD, EUR, etc.)
- **Start Date**: Contract start date (YYYY-MM-DD format)
- **End Date**: Contract end date (YYYY-MM-DD format)
- **Payment Terms**: Payment terms (e.g., "30 days", "60 days", "90 days")
- **Payment Milestones**: Payment milestones (e.g., "30% on signing")
- **Contract Type**: "Domestic" or "International"

## Code Structure

### Main Functions

1. **`extract_contract_data_ai()`** - AI-powered extraction using Azure OpenAI
2. **`extract_contract_data_fallback()`** - Pattern matching fallback
3. **`extract_contract_data()`** - Main function that tries AI first, then falls back

### Files Modified

- `cash_flow_calculator.py` - Core extraction logic with AI integration
- `app.py` - Streamlit UI imports and uses the extraction functions
- `requirements.txt` - Added Langchain and Azure OpenAI dependencies

## Dependencies

New dependencies added:
- `langchain>=0.1.0`
- `langchain-openai>=0.0.5`
- `azure-identity>=1.15.0`
- `python-dotenv>=1.0.0`

## Testing

The existing unit tests continue to work because:
- Tests use the fallback pattern matching when AI is unavailable
- The `extract_contract_data()` function automatically handles fallback
- No changes needed to test files

## Usage

The AI extraction is automatically used when:
1. Langchain is installed
2. Azure OpenAI credentials are configured in `.env`
3. Contract PDFs are uploaded through the Streamlit UI

The system will automatically use AI extraction if available, otherwise falls back to pattern matching.

## Benefits

1. **Better Accuracy**: AI understands context and extracts information more accurately
2. **Handles Variations**: Works with different contract formats and structures
3. **Extracts Complex Fields**: Can identify payment milestones and complex terms
4. **Graceful Degradation**: Falls back to pattern matching if AI is unavailable
5. **No Breaking Changes**: Existing functionality continues to work

## Future Enhancements

- Support for multi-page contracts
- Batch processing of multiple contracts
- Custom prompt templates for specific contract types
- Confidence scores for extracted fields
- Manual review flagging for uncertain extractions

