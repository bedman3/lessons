

```python
_h -> history of ticks
bid_0_nbbo_price_h
bid_0_nbbo_qty_h
ask_0_nbbo_price_h
ask_0_nbbo_qty_h
local_timestamp_h
exchange_timestamp_h
event_price_h
event_fill_price_h
last_trade_price_h
bid_ask_crossed_h (bool) = bid_0_price_h >= ask_0_price_h
uncrossed_spared_h = np.nan if bid_ask_crossed_h else ask_0_nbbo_price_h - bid_0_nbbo_price_h
uncrossed_min_spread = min_elem(uncrossed_spread_h)
uncrossed_mid_h = np.nan if bid_ask_crossed_h else mid(ask_0_nbbo_price_h, bid_0_nbbo_price_h)
uncrossed_mid = last(uncrossed_mid_h)

uncrossed_spread_frac_h = uncrossed_spread_h / uncrossed_mid_h
uncrossed_min_spread_frac = min_elem(uncrossed_spread_frac_h)
vol_traded_h
vol_placed_h
is_dark_h
trade_tick_mask_h = vol_traded_h > 0 & not is_dark_h & not auction trade
t_price_h = filter(trade_tick_mask_h, event_fill_price_h)
t_quantity_h = filter(trade_tick_mask_h, vol_traded_h)
t_bid_0_nbbo_price_h = filter(trade_tick_mask_h, bid_0_nbbo_price_h)
t_ask_0_nbbo_price_h = filter(trade_tick_mask_h, ask_0_nbbo_price_h)
```