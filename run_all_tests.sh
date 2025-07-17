#\!/bin/bash

# Initialize master log
echo "=== Comprehensive Test Execution Report ===" > test_master_results.log
echo "Started: $(date)" >> test_master_results.log
echo "=========================================" >> test_master_results.log

# Counter variables
total_tests=0
successful_tests=0
failed_tests=0
timeout_tests=0

# Function to run a single test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file")
    local log_file="logs/${test_name%.py}.log"
    
    echo "" >> test_master_results.log
    echo "[$test_name]" >> test_master_results.log
    echo "Started: $(date)" >> test_master_results.log
    
    # Create logs directory
    mkdir -p logs
    
    # Run with timeout
    source .venv/bin/activate && timeout 600 python "$test_file" > "$log_file" 2>&1
    local exit_code=$?
    
    ((total_tests++))
    
    if [ $exit_code -eq 0 ]; then
        echo "Status: SUCCESS" >> test_master_results.log
        ((successful_tests++))
    elif [ $exit_code -eq 124 ]; then
        echo "Status: TIMEOUT (10 minutes)" >> test_master_results.log
        ((timeout_tests++))
    else
        echo "Status: FAILED (exit code: $exit_code)" >> test_master_results.log
        ((failed_tests++))
    fi
    
    # Extract summary info
    if [ -f "$log_file" ]; then
        # Basic stats
        lines=$(wc -l < "$log_file")
        echo "Log size: $lines lines" >> test_master_results.log
        
        # Extract key metrics
        echo "Key findings:" >> test_master_results.log
        grep -i "total.*segment\ < /dev/null | extracted.*character\|segment.*count" "$log_file" 2>/dev/null | head -3 | sed 's/^/  - /' >> test_master_results.log || true
        grep -i "visual.*element\|vlm.*process\|image.*found" "$log_file" 2>/dev/null | head -2 | sed 's/^/  - /' >> test_master_results.log || true
        grep -i "success\|passed\|completed" "$log_file" 2>/dev/null | tail -2 | sed 's/^/  - /' >> test_master_results.log || true
    fi
}

# Run all tests
for test_file in tests/*.py tests/debugging/*.py tests/debugging/*/*.py; do
    if [ -f "$test_file" ] && [[ \! "$test_file" == *"__pycache__"* ]]; then
        run_test "$test_file"
    fi
done

# Final summary
echo "" >> test_master_results.log
echo "=========================================" >> test_master_results.log
echo "FINAL SUMMARY" >> test_master_results.log
echo "=========================================" >> test_master_results.log
echo "Total tests: $total_tests" >> test_master_results.log
echo "Successful: $successful_tests" >> test_master_results.log
echo "Failed: $failed_tests" >> test_master_results.log
echo "Timeouts: $timeout_tests" >> test_master_results.log
echo "Completed: $(date)" >> test_master_results.log

