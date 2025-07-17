#!/usr/bin/env python3
import json

with open('tests/debugging/pixtral_test_results.json', 'r') as f:
    data = json.load(f)
    
print('\n=== VLM Comparison Results ===')
for result in data:
    print(f"\nPage {result['page']}:")
    for analysis in result['analyses']:
        print(f"\n  {analysis['model']}:")
        print(f"    Success: {analysis['success']}")
        print(f"    Confidence: {analysis.get('confidence', 0):.0%}")
        if analysis['success']:
            print(f"    Description: {analysis['description'][:150]}...")
        else:
            print(f"    Error: {analysis.get('error_message', 'Unknown error')}")