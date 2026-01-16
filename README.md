# Licensing Cash Flow Calculator

A Streamlit application for calculating licensing cash flow projections based on multiple input sources.

## Features

This application follows the flow outlined in the three CSV files:

1. **Revenue File Processing** - Validates and processes channel-wise revenue breakdown
2. **Contract PDF Processing** - AI-powered extraction of contract terms and calculation of accrual schedules
3. **Billing Pattern Calculation** - Projects monthly billing based on revenue and weighted averages
4. **Collection Trend Analysis** - Analyzes historical collection patterns from aging reports
5. **OS Collection Application** - Applies collection trends to outstanding balances
6. **Cash Flow Projection** - Generates comprehensive monthly cash flow forecasts

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Input Files Required

1. **Revenue File** (CSV/Excel) - Channel-wise breakdown
2. **Contract PDFs** - Client agreements with payment terms
3. **Aging Report** (CSV/Excel) - Outstanding balances by aging buckets
4. **Invoice Reports** (CSV/Excel) - Collection records
5. **Month-end Report** (CSV/Excel) - Closing data
6. **AR Report** (CSV/Excel) - Receivables data

## Usage Flow

1. **Upload Files**: Upload all required input files
2. **Revenue Processing**: System validates format and extracts channel breakdown
3. **Contract Processing**: AI extracts contract data and calculates accrual schedules
4. **Billing Pattern**: Calculates monthly billing projections with GST adjustments
5. **Collection Trend**: Analyzes historical collection patterns
6. **Cash Flow Projection**: Generates final cash flow forecast with AR movement

## Output

The application generates:
- Monthly billing projections (Domestic/International split)
- Collection projections based on historical trends
- AR movement tracking
- Net cash flow projections
- Downloadable CSV report

## Testing

The project includes comprehensive unit tests covering all major functions:

### Running Tests

**Using the test runner script:**
```bash
./run_tests.sh
```

**Using unittest directly:**
```bash
python -m unittest test_cash_flow -v
```

**Using pytest (if installed):**
```bash
pytest test_cash_flow.py -v
```

### Test Coverage

- ✅ PDF extraction and contract data processing (4 tests)
- ✅ Revenue file processing and validation (2 tests)
- ✅ Accrual schedule calculation with GST logic (2 tests)
- ✅ Billing pattern calculation (2 tests)
- ✅ Collection trend analysis (2 tests)
- ✅ OS collection trend application (1 test)
- ✅ Actual cash flow consolidation (2 tests)
- ✅ Cash flow projection calculation (2 tests)
- ✅ End-to-end integration flow (1 test)

**Total: 18 tests - All passing ✅**

See `TEST_SUMMARY.md` for detailed test documentation.

### Test Data

Mock input files for testing are located in the `test_data/` directory:
- `revenue_file.csv` - Sample revenue data
- `aging_report.csv` - Sample aging report
- `invoice_report.csv` - Sample invoice data
- `month_end_report.csv` - Sample month-end data
- `ar_report.csv` - Sample AR data

The tests also use `example_contract.pdf` for PDF extraction testing.

## AI-Powered Contract Extraction

The application now uses **Azure OpenAI with Langchain** for intelligent contract data extraction:

- **AI Extraction**: Automatically extracts contract fields (Client Name, Deal Value, Dates, Payment Terms, etc.) using Azure OpenAI
- **Automatic Fallback**: Falls back to pattern matching if AI is unavailable
- **Configuration**: Uses `.env` file for Azure OpenAI credentials (see `AI_INTEGRATION.md` for details)

### Setup for AI Extraction

1. Ensure `.env` file contains:
   ```
   AZURE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_DEPLOYMENT=gpt-4
   ```

2. Install AI dependencies:
   ```bash
   pip install langchain langchain-openai python-dotenv
   ```

See `AI_INTEGRATION.md` for detailed documentation.

## Notes

- Contract PDF extraction uses AI (Azure OpenAI) with pattern matching fallback
- Collection trends are calculated from historical aging data
- GST (18%) is applied to Domestic transactions only
- TDS/WHT adjustments are made at 10% (configurable)
- Core calculation functions are extracted in `cash_flow_calculator.py` for testability

