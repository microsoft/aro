# MICROSOFT RESPONSIBLE AI TRANSPARENCY DOCUMENTATION FOR RESEARCH

## Adaptive Rotation Optimizers (ARO)

### Overview

Adaptive Rotation Optimizers (ARO) is a Python library providing a family of Pytorch optimizers for training LLMs.

These optimizers speed up training, compared to vanilla AdamW and other baselines, by applying rotations to matrix parameters and optimizing them in the rotated space.

### What can ARO do

ARO was developed to train small and middle sized LLMs (from 100M to 10Bs sized models) supporting distributed training schemes (DDP and FSDP).

ARO consists of optimizers that process gradients in a specific way—using rotations—to optimize a loss or solve a task. For example, during pretraining, this might involve minimizing the loss for next-token prediction. In a post-training setting, such as a reasoning application (e.g., solving math problems), the task could include generating multiple candidate responses, ranking them, and guiding learning toward the best answers. ARO only requires the ability to backpropagate—i.e., compute gradients—for the specified loss or task and remains agnostic to the nature of the tasks themselves.

A detailed discussion of ARO, including how it was developed and tested, can be found in our paper at: <https://arxiv.org/abs/2602.09006>.

### Intended Uses

ARO is best suited for optimization of LLMs, including pre-training and finetuning.

ARO is being shared with the research community to facilitate reproduction of our results and foster further research in this area.

ARO is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

### Out-of-scope uses

ARO may not be well suited for other types of optimization problems.

We do not recommend using ARO in commercial or real-world applications without further testing and development. It is being released for research purposes.

ARO was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.  
Without further testing and development, ARO should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.

We do not recommend using ARO in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).

## How to get started

To begin using ARO, follow instructions given in <https://github.com/microsoft/aro>.

## Evaluation

ARO was evaluated on its ability to pre-train different LLMs, including dense and MoE architectures, such as GPT2, NanoCHAT and QWEN3.

A detailed discussion of our evaluation methods and results can be found in our paper at: <https://arxiv.org/abs/2602.09006>.

#### Evaluation Methods

We used cross-entropy loss to measure ARO’s performance in the pre-training task.

We compared the performance of ARO against baseline methods like AdamW evaluating wall-clock optimization time and number of steps to attain a certain loss.

The model trained for evaluation was GPT2. For more on this specific model, please see <https://github.com/karpathy/nanoGPT>.

Results may vary if ARO is used with a different model based on its unique design, configuration and training.

#### Evaluation Results

At a high level, we found that ARO performed 2x-3x speed gains in research-scale models, and x1.36 in frontier models.

## Limitations

ARO was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.

ARO has primarily been tested on transformer-based language models using PyTorch. Its performance on other model types, frameworks, or non-standard training pipelines has not been extensively evaluated.

ARO performance and convergence behavior may vary across hardware platforms (NVIDIA, AMD) and distributed setups (DDP, FSDP, ZeRO, etc.). Some ARO components rely on collective communication primitives whose efficiency and numerical behavior depend on backend implementation and cluster configuration.

ARO introduces additional computation and communication overhead to the optimization step during training, where communication generally overlaps computation (in distributed schemes). We have benchmarked ARO on models ranging from small to mid-sized (~100M-10B parameters), consistently observing speed-up gains compared to AdamW. For larger-scale models, careful profiling may be necessary to fully understand performance trade-offs.

Training outcomes may vary due to random initialization, nondeterministic GPU kernels, and distributed reduction ordering. Reproducibility across runs or hardware should be verified independently.

While ARO itself does not encode linguistic or societal bias, any biases or artifacts present in the training data may be propagated or amplified through training. Users remain responsible for evaluating the fairness and safety of trained models.

Outputs generated by AI models trained with ARO may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.

There has not been a systematic effort to ensure that systems using ARO are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate.

## Best Practices

Better performance can be achieved when using ARO for training LLMs by optimizing 2D parameters with ARO, and 1D parameters, LLM head and embedding layers with AdamW. We provide examples on how to separate parameters for faster optimization in <https://github.com/microsoft/aro>.

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.

Developers should follow transparency best practices and inform end-users they are interacting with an AI system.

## License

MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Contact

This research was conducted by members of <https://www.microsoft.com/en-us/research/>.  
We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please create an issue at the repository.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.
