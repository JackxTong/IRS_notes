- **Order size (log-transform)**  
  - Use `log(order_size_dv01 + 1)` instead of raw DV01.  

- **Market liquidity / volatility**  
  - Mid-market **bid–ask spread** for the trade’s tenor.  
  - **Curve slope/curvature** measures (e.g., 2s10s, 5s30s).  
  - **Implied rate volatility** (swaption ATM vol) for the relevant tenor.  

- **Temporal features**  
  - **Time of day** (trading hour, market open/close, fixing times).  
  - **Day of week**.  
  - **Month-end / quarter-end indicator**.  

- **Dealer competition / concentration**  
  - **Herfindahl–Hirschman Index (HHI)** of dealer quote sizes.  
  - **Best-to-median quote spread** (dispersion across dealer quotes).  

- **Market microstructure dynamics**  
  - **Recent trade direction** (sign of last N trades).  
  - **Recent spread trend** (rolling average or volatility of spreads).  

- **Client / order characteristics** (if available)  
  - **Client type / tier** (e.g., top-tier vs others).  
  - **Order urgency / RFQ size bucket**.  
  - **Client hit ratio** (historical trade acceptance rate).  

- **Nonlinearities & interactions**  
  - Squared tenor term (`tenor^2`).  
  - Interaction: **order size × imbalance**.  
  - Interaction: **tenor × num_dealers**.  