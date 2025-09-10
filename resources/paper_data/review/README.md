# Review

In the file `papers.csv` the result of an advanced SCOPUS query is contained. Each entry is additionally marked with the reasons for exclusion from comparison.

### Query

The executed SCOPUS query is:

```
TITLE-ABS-KEY ( ( ( "automatic" OR "automated" OR "generation" OR "synthesis" OR "tool" OR "framework" OR "evaluation" OR "configuration" OR "guideline" OR "methodology" OR "benchmark" ) AND ( "load test*" OR "stress test*" OR "performance test*" OR "workload generation" OR "benchmark generation" OR "performance evaluation" OR "autoscaler" ) ) AND ( "microservice*" OR "auto-scal*" OR "auto scaling" OR "elastic system" OR "elastic microservice*" OR "autoscal*" ) ) AND ( LIMIT-TO ( SUBJAREA,"COMP" ) ) AND ( LIMIT-TO ( DOCTYPE,"cp" ) OR LIMIT-TO ( DOCTYPE,"ar" ) ) AND ( LIMIT-TO ( LANGUAGE,"English" ) )
```

### Results

All papers were marked with reasons for exclusions. Studies close enough to ours are marked with more details and are generally discussed within the paper itself.