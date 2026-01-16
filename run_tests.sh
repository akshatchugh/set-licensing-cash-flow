#!/bin/bash
# Test runner script for cash flow calculator

echo "Running Cash Flow Calculator Tests..."
echo "====================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found. Make sure dependencies are installed."
fi

echo ""
echo "Running unit tests..."
echo ""

# Run tests
python -m unittest test_cash_flow -v

echo ""
echo "====================================="
echo "Tests completed!"

