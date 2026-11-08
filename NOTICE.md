# Source and modification notice

Parts of `core/`, including the AVP controller and video utilities, derive from
[SalesforceAIResearch/ActiveVideoPerception](https://github.com/SalesforceAIResearch/ActiveVideoPerception).
The upstream Creative Commons Attribution-NonCommercial 4.0 International
license is reproduced without modification in `LICENSE.txt`.

PAVA modifications include persistent local retrieval, local audit, the
observation-only handoff, transcript/index integration, and associated runtime
support. The portable handoff additionally parameterizes the model endpoint,
model name and initial frame count, and adds preparation, bounded batch
execution, API accounting and reporting. It retains the source implementation's
existing prompts and control flow rather than claiming to be an unmodified AVP
release.

Benchmark question data and videos retain their respective dataset terms.
The download script retrieves videos from the sources identified in its plan;
no video files or model weights are redistributed in this repository.
