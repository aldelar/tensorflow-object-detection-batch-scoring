# tensorflow-object-detection-batch-scoring

Azure ML batch scoring sample project leveraging:
- Azure ML Pipelines
- Azure ML Compute (dynamic on-demand clusters for training and scoring)
- Pipelines published as API REST endpoint

This solution enables a serveless approach to complex scoring scenarios (potential data preprocessing, quality control + scoring + post processing steps) leveraring different compute nodes for each job (cpu optimized, memory optimized, gpu, etc.)