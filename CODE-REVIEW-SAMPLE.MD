# Interview Code Review Exercise

## SNIPPET 1: Math Calculations with Pandas DataFrames

### Requirement:
Calculate portfolio momentum metrics for a stock universe. The function should:
1. Load stock price data from a SQL database for a date range
2. Calculate 1-month, 3-month, and 6-month price momentum for each stock
3. Calculate weighted average momentum (50% 1-month, 30% 3-month, 20% 6-month)
4. Aggregate by sector to compute sector-level momentum
5. Return a DataFrame with columns: `date`, `ticker`, `sector`, `momentum`, `sector_momentum`

### Code Snippet:

```python
import pandas as pd
import sqlalchemy
from datetime import datetime, timedelta

def calculate_portfolio_momentum(universe: str, from_date: datetime, to_date: datetime):
    """
    Calculate momentum metrics for stocks in the given universe.
    """
    engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/oracdb')
    
    # Load data
    query = """
        SELECT date, ticker, share_price, universe
        FROM stock_entry
        WHERE universe = %s
        AND date >= %s
        AND date <= %s
        ORDER BY ticker, date
    """
    df = pd.read_sql(query, engine, params=(universe, from_date, to_date))
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Calculate momentum
    df['price_1m'] = df['share_price'] / df['share_price'].shift(30) - 1
    df['price_3m'] = df['share_price'] / df['share_price'].shift(90) - 1
    df['price_6m'] = df['share_price'] / df['share_price'].shift(180) - 1
    
    # Weighted momentum
    df['momentum'] = df['price_1m'] * 0.5 + df['price_3m'] * 0.3 + df['price_6m'] * 0.2
    
    # Get sector data
    sector_query = """
        SELECT ticker, sector
        FROM index_constituents
        WHERE universe = %s
    """
    sector_df = pd.read_sql(sector_query, engine, params=(universe,))
    
    # Merge sector data
    df = df.merge(sector_df[['ticker', 'sector']], on='ticker', how='left')
    
    # Calculate sector average momentum
    sector_momentum = df.groupby(['date', 'sector'])['momentum'].mean()
    df['sector_momentum'] = df.apply(
        lambda row: sector_momentum.loc[row['date'], row['sector']],
        axis=1
    )
    
    result = df[['date', 'ticker', 'sector', 'momentum', 'sector_momentum']]
    return result
```

---

## SNIPPET 2: Data Pipeline

### Requirement:
Build a data pipeline that:
1. Loads factor data from a SQL database for multiple factor IDs and a date range
2. Validates that all required factors are present for each date/ticker combination
3. Calculates a composite factor score by averaging the normalized factor values
4. Merges the composite score with existing index data
5. Saves the results back to the database
6. Logs summary statistics

### Code Snippet:

```python
import pandas as pd
import numpy as np
import sqlalchemy
from datetime import datetime

def process_factor_pipeline(universe: str, factor_ids: list, from_date: datetime, to_date: datetime):
    """
    Process factor data pipeline: load, validate, calculate, and save composite factors.
    """
    engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/oracdb')
    
    # Load factor data
    all_factors = []
    for factor_id in factor_ids:
        query = """
            SELECT date, ticker, value, factor_id
            FROM factor_data
            WHERE universe = %s
            AND factor_id = %s
            AND date >= %s
            AND date <= %s
        """
        df = pd.read_sql(query, engine, params=(universe, factor_id, from_date, to_date))
        df['factor_id'] = factor_id
        all_factors.append(df)
    
    if len(all_factors) == 0:
        print("No factor data found")
        return
    
    combined_df = pd.concat(all_factors, ignore_index=True)
    
    # Validate required factors present
    date_ticker_combos = combined_df.groupby(['date', 'ticker']).size()
    missing_factors = date_ticker_combos[date_ticker_combos < len(factor_ids)]
    if len(missing_factors) > 0:
        print(f"Warning: {len(missing_factors)} date/ticker combinations missing factors")
    
    # Normalize factor values
    for factor_id in factor_ids:
        factor_data = combined_df[combined_df['factor_id'] == factor_id]
        mean_val = factor_data['value'].mean()
        std_val = factor_data['value'].std()
        combined_df.loc[combined_df['factor_id'] == factor_id, 'normalized'] = \
            (combined_df.loc[combined_df['factor_id'] == factor_id, 'value'] - mean_val) / std_val
    
    # Calculate composite score
    composite = combined_df.groupby(['date', 'ticker'])['normalized'].mean()
    composite_df = composite.reset_index()
    composite_df.columns = ['date', 'ticker', 'composite_score']
    
    # Load and merge index data
    index_query = """
        SELECT date, ticker, sector, sub_sector
        FROM index_constituents
        WHERE universe = %s
        AND date >= %s
        AND date <= %s
    """
    index_df = pd.read_sql(index_query, engine, params=(universe, from_date, to_date))
    
    final_df = index_df.merge(composite_df, on=['date', 'ticker'], how='inner')
    
    # Save results
    final_df['universe'] = universe
    final_df[['date', 'ticker', 'universe', 'composite_score']].to_sql(
        'composite_factors',
        engine,
        if_exists='append',
        index=False
    )
    
    # Log statistics
    print(f"Processed {len(final_df)} records")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Average composite score: {final_df['composite_score'].mean()}")
    
    return final_df
```

## Project technical discussion ( Stock Screener ) 
1. Raw data: Company data ticker/symbol, market data, fundamentals, estimates ( 50 different types) for S&P500, for the last 20 years
2. Building numerical factors/metrics library 
3. Ranking stocks based on selected factors
4. Back testing
