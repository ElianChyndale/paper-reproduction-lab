# Limitations

- The release uses only synthetic fixtures. Results do not estimate performance
  on original paper datasets or real applications.
- The retrieval study implements TF-IDF plus TruncatedSVD, not a trained dense
  passage retriever. Its feature reranker is not a neural cross-encoder.
- BM25 tokenization and document processing are deliberately minimal and do
  not reproduce Lucene/Anserini pipelines or paper-specific preprocessing.
- The calibration study uses generated binary logits. It does not train or
  evaluate neural networks, reproduce image/document datasets, or test
  paper-reported architectures and tables.
- ECE depends on binning; this release fixes ten equal-width confidence bins.
  Small synthetic samples can make ECE and risk-coverage values unstable.
- Platt and temperature parameters are fixture-specific and must not be reused
  as production calibration parameters.
- The workflow and agent methods are deterministic state-machine proxies. They
  are not LLM agents, do not reproduce ReAct or AgentBench, and cannot support
  claims about real single-agent or multi-agent systems.
- Unsafe actions are names in a fixture. No tool or external side effect is
  executed, so the study does not measure runtime containment.
- The fixed seed demonstrates repeatability for one configured sample; it does
  not quantify variance across seeds.
- No statistical significance, confidence interval, power analysis, external
  validity, or robustness to distribution shift is claimed.
- Passing schema, metric, test, wheel, and CI checks proves only the documented
  contracts and bounded fixtures.
