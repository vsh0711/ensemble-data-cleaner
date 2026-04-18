 AI-powered data quality detector using ensemble ML methods.
    
    Three methods vote on whether each row is anomalous:
      1. Isolation Forest  — detects global outliers via random partitioning
      2. Z-score           — catches univariate extreme values statistically  
      3. Local Outlier Factor — detects local density anomalies (contextual outliers)
________________________________________________________________________________________
The tool will do all three things — preloaded demo data so anyone can instantly see it working, plus CSV upload for their own data, plus a downloadable clean dataset.
_______________________________________________________________________________________   
A record is flagged only when 2 or more methods agree.
    - This consensus approach dramatically reduces false positives.

Returns df with four new columns appended:
    iso_flag    — 1 if Isolation Forest says anomaly
    zscore_flag — 1 if any feature exceeds zscore_threshold std devs
    lof_flag    — 1 if Local Outlier Factor says anomaly
    anomaly     — 1 if 2+ methods agree (ensemble decision)
    nomaly_score — continuous risk score (higher = more anomalous

Why three methods instead of one?
 1. Isolation Forest is excellent at finding global outliers — records that are unusual compared to the entire dataset. 
 2. Z-score catches simple univariate extremes — a single feature value that is 4 standard deviations from the mean. 
 3. Local Outlier Factor finds contextual anomalies — records that look normal globally but are unusual within their local neighbourhood. 
 Each method has blind spots. The consensus rule means flag only when something  at least two independent methods agree — this is the ensemble principle applied to data cleaning, not just prediction.