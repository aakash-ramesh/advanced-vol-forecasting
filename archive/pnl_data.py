import pandas as pd
import numpy as np

parent_dir = "/Users/aramesh/Documents/GitHub/advanced-vol-forecasting/"
opt_data = pd.read_parquet(parent_dir+"data/raw/opprcd2022_spy.parquet")
secprc = pd.read_parquet(parent_dir+"data/raw/secprc2022_spyx.parquet")

opt_data = opt_data[[
    'secid', 'date', 'symbol', 'cp_flag', 'strike_price',
    'exdate', 'best_bid', 'best_offer', 'volume',
    'contract_size', 'cfadj']]

idx = {"SPY": 109820, "SPX": 108105}

spy_data = secprc[secprc['secid']==109820][['date', 'low', 'high', 'close', 'volume', 'return', 'cfadj',
       'open', 'cfret', 'shrout']].reset_index()


spy_data['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * (np.log(spy_data['high']/spy_data['low']))**2)
spy_data['gk_vol'] = np.sqrt(0.5*(np.log(spy_data['high']/spy_data['low']))**2 - (2*np.log(2)-1)*(np.log(spy_data['close']/spy_data['open']))**2)

spy_data = spy_data[['date', 'low', 'high', 'close', 'volume', 'return', 'cfadj',
       'open', 'cfret', 'shrout', 'parkinson_vol', 'gk_vol']]

final_pnl_opt = opt_data.merge(spy_data, on='date', how='left')

# print(final_pnl_opt)

final_pnl_opt.to_parquet(parent_dir+"data/master/pnl_data_spy.parquet", index=False)