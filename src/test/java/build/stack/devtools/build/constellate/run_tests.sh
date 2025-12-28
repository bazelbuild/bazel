#!/bin/bash
# Comprehensive test suite for constellate tool

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDATA_DIR="$SCRIPT_DIR/testdata"
OUTPUT_DIR="/tmp/constellate_test_output"

# Build constellate
echo -e "${YELLOW}Building constellate...${NC}"
bazel build //src/main/java/build/stack/devtools/build/constellate:constellate

CONSTELLATE="$(bazel info bazel-bin)/src/main/java/build/stack/devtools/build/constellate/constellate"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local input_file="$2"
    local output_file="$OUTPUT_DIR/${test_name}.pb"

    echo -e "${YELLOW}Running test: ${test_name}${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))

    if "$CONSTELLATE" --input="$input_file" --output="$output_file" 2>&1; then
        if [ -f "$output_file" ]; then
            echo -e "${GREEN}✓ Test passed: ${test_name}${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))

            # Show some basic validation
            echo "  Output file size: $(wc -c < "$output_file") bytes"

            # Try to decode and show entity counts if protoc is available
            if command -v protoc &> /dev/null; then
                echo "  Attempting to decode protobuf..."
                # Note: This would need the actual proto descriptor
            fi
        else
            echo -e "${RED}✗ Test failed: ${test_name} (no output file)${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        echo -e "${RED}✗ Test failed: ${test_name} (execution error)${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

echo "========================================="
echo "Constellate Test Suite"
echo "========================================="
echo ""

# Run tests
run_test "simple_test" "$TESTDATA_DIR/simple_test.bzl"
run_test "comprehensive_test" "$TESTDATA_DIR/comprehensive_test.bzl"

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Tests run:    $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
else
    echo -e "Tests failed: $TESTS_FAILED"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
