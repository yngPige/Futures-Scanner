"""
Simple test script to verify 5 decimal place formatting.
"""

def main():
    """Main function."""
    print("Testing 5 Decimal Place Formatting")
    print("=================================")
    
    # Test values
    values = [
        0.12345,
        1.23456,
        12.34567,
        123.45678,
        1234.56789
    ]
    
    # Format with 5 decimal places
    print("\nFormatting values with 5 decimal places:")
    for value in values:
        formatted = f"{value:.5f}"
        print(f"Original: {value} -> Formatted: {formatted}")
        
        # Verify decimal places
        if '.' in formatted and len(formatted.split('.')[1]) == 5:
            print(f"✓ Value has 5 decimal places")
        else:
            print(f"✗ Value does NOT have 5 decimal places")
        print()
    
    print("Test completed.")

if __name__ == "__main__":
    main()
