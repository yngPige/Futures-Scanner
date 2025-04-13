"""
Test script to save a terminal chart to a file.

This script creates a simple chart and saves it to a file.
"""

import plotext as plt
import numpy as np
from datetime import datetime, timedelta

def main():
    """Main function."""
    print("Testing saving terminal chart to file")

    # Generate sample data
    n_points = 50
    today = datetime.now()
    dates = [(today - timedelta(days=n_points-i)).strftime('%d/%m/%Y') for i in range(n_points)]

    # Generate price data with some randomness
    base_price = 70000  # Base price for BTC
    prices = [base_price + np.random.normal(0, 1000) for _ in range(n_points)]

    # Set date format
    plt.date_form('d/m/Y')

    # Clear the previous plot
    plt.clf()

    # Set the title and labels
    plt.title("BTC/USDT Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Plot the price data
    plt.plot(dates, prices, label="BTC Price")

    # Add a horizontal line at the average price
    avg_price = sum(prices) / len(prices)
    plt.hline(avg_price, color="red")

    # Print the average price
    print(f"Average price: {avg_price:.2f}")

    # Add legend
    plt.legend()

    # Save the plot to a file
    plt.save_fig("terminal_chart.txt")

    # Show the plot
    plt.show()

    print("Test completed successfully")

if __name__ == "__main__":
    main()
