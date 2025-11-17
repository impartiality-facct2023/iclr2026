# Reviews for: OptiFluence: Scalable and Principled Design of Privacy Canaries
- ### Reviewer_VkF8
	- **Rating:** 4
	- **Confidence:** 3
	- #### Summary
	    The paper proposes a framework for constructing privacy canaries—artificial samples used to audit data leakage in machine learning models. Unlike prior methods that rely on mislabeled or out-of-distribution samples, OptiFluence formulates canary design as a bilevel optimization problem that maximizes the likelihood-ratio statistic used in membership inference attacks, effectively identifying samples that most alter the model’s behavior when included in training. The framework combines two key components: IF-Init, which uses influence functions to pre-select rare, highly self-influential samples likely to be memorized, and Unrolled-Opt, which refines these candidates by differentiating through the training process using memory-efficient techniques such as rematerialization and truncated backpropagation through time (TBPTT). Experiments on MNIST and CIFAR-10 demonstrate that OptiFluence achieves near-perfect detectability (up to 99.8% TPR at 0.1% FPR), outperforming baselines by up to 415×.
	- #### Scores
		- Soundness: 2
		- Presentation: 3
		- Contribution: 2
	- #### Strengths
	    1. The proposed OptiFluence pipeline combines influence-based initialization (IF-Init) with unrolled optimization enhanced by rematerialization and truncated backpropagation (ReMat+TBPTT) for scalability.
	    
	    2. Achieves near-perfect canary detectability (up to 99.8% TPR at 0.1% FPR) and up to 415× improvement over heuristic baselines on MNIST and CIFAR-10.
	    
	    3. Optimized canaries generalize well across architectures (e.g., ResNet-9 → ResNet-50), enabling efficient third-party or regulatory auditing without model retraining.
	    
	    4. Includes detailed ablation studies, comparisons to prior methods (e.g., metagradient-based optimization), and evaluations under DP-SGD training.
	    
	    5. The paper is well-structured, with a clear presentation of motivation, methodology, and results.
	- #### Weaknesses
	    1. The files at the anonymous link do not open. Please either include a zip file with the source code and scripts in the supplementary material or update the link to ensure the files are accessible.
		- We apologize for the link not working. We have re-uploaded and regenerated the link to the anonymous repository that contains the code.
	- 2. The central idea—to optimize the log-likelihood ratio via a bilevel objective—is elegant but remains largely heuristic. The paper argues that maximizing the difference in logits between models trained with and without the canary approximates the true likelihood-ratio test (Equation 5), yet provides no formal proof that this surrogate objective is consistent or unbiased. Without theoretical guarantees or approximation bounds, it is unclear whether the optimized canary truly maximizes detectability in general or only within the specific experimental setup.
		- We largely agree with the reviewer. However, the lack of general bounds is expected given the bi-level characterization of the problem with the training loss objective (of a multi-million parameter neural network) in the constraint set. Even the state-of-the-art optimization results for neural networks are limited to a few layer networks. The reviewer may argue that a formal result might be possible for small linear models (logistic regression); we would counter then that prior work <CITE> have shown that such small models typically do not represent 
		   the memorization and privacy risks that canaries are used to audit.
		- Our work empirically demonstrates the practical feasibility of optimized canaries as a framework. We do agree formal results would be intriguing but given the lack of relevant literature, and the density of the formalization and modeling already present in the paper; we intend to pursue formal results in future work.
	- 3.  The bilevel objective requires differentiating through the entire training trajectory, but the paper does not discuss convergence, stability, or variance of this optimization—particularly when truncated backpropagation and rematerialization are used. The claim in Section 6.3 that unrolled optimization yields “exact gradients” appears overstated, since ReMat+TBPTT introduces gradient truncation and therefore only provides approximate updates. It would strengthen the paper to quantify how this approximation impacts final canary detectability.
		- We do not use the term "exact gradient" lightly. Our "unrolled" basedline on MNIST does indeed take a hyper-gradient that is exact in the sense that we create the complete training run as a single computational graph and differentiate through it using automatic differentiation. This is clearly not scalable, but this is as exact a gradient as weight gradients.
		- To respond to the reviewer point about the impact of truncation, we have devised and ran an additional experiment where we change the length of the backpropagation.
		- TODO ==@Arielle== Can you run this experiment?
	- 4.  While the paper references differential privacy definitions and ε-bounds (Equation 2), it never formally connects the empirical detectability metric (TPR@FPR) to theoretical privacy parameters ε or δ. The framework is therefore empirical rather than analytical, and the paper should make this distinction explicit.
		- Prior work has done this already as empirical lower bounds such as [[Bayesian Estimation of Differential Privacy]]
		- No particular novelty to out work. As we show direct LIRA attack scores instead of going through the proxy of epsilon lower bounds.
		- In addition, new work [[Attack-Aware Noise Calibration for Differential Privacy]] shows that privacy risks and mitigations can be formalized directly within a privacy attack framing. Thus a characterization in terms of epsilon values is useful but not essential.
	- 5.  Because OptiFluence directly maximizes the LiRA hinge-based likelihood-ratio score, the optimized canaries may overfit to this particular attack formulation. The paper does not evaluate the canaries under alternative membership inference metrics (e.g., confidence-, entropy-, or loss-based scores), leaving open the question of whether detectability generalizes to unseen auditing methods.
		- TODO ==@Florian== you can probably answer this better
		- Note that our evaluation are done completely independent of our optimization. Unlike first party privacy audits, we just need to make sure that out canaries are sampled during training. Otherwise we use a standard privacy auditing framework.
		- We cannot scientifically discuss "unseen auditing methods" of the future but our adversary model is that of the privacy adversary attempting to distinguish. Any future auditing method  that adopts this adversary model is benefit from our canaries.
	- 6. The reported transferability of canaries across architectures (e.g., ResNet-9 → ResNet-50) is intriguing but lacks theoretical explanation. The paper attributes it qualitatively to shared representation geometry, yet provides no analysis of why optimized samples remain highly distinguishable under different model dynamics. This weakens the claim that OptiFluence supports “third-party audits without retraining.”
		- TODO @Nicolas/@Florian can you help here?
		- We do not make any theoretical claim about transferability but we agree that this is an exciting venue for future work.
		- We validate our transferability claim empirically. To the best of our knowledge, in prior cases of   transferability (such as adversarial examples), the community have not been able to characterize a theoretical reasoning for this phenomenon.
	- 7. Figures 4 and 6 show that optimized canaries—particularly on CIFAR-10—often appear visually unnatural or out-of-distribution. This raises the possibility that their detectability stems from atypical low-level statistics rather than genuine memorization.
		- TODO We need to tackle the memorization angle with care. It is used and abused in the literature. So, we use it as well. We can for example say we will ensure to clarify what we mean by memorization is purely from a privacy attack vulnerability. ==@Nicolas== makes sense?
		- To the best of our knowledge, "Memorization" unlike differential privacy does not have an agreed-upon definition which precludes a rigorous claim on what constitutes "genuine memorization."
		- A canary, under the DP definition, does not even need to come from a particular data distribution. Therefore, our canaries fit the definition of the privacy canary.
		- We understand that it might be desirable to add a constraint on the "naturalness," or what is often done in adversarial examples, to bound the amount of perturbation. But as we have said, given the fact that any sample
		- Finally, we note that prior work in privacy auditing uses far more unnatrual-looking canaries. See <CITE> where a square pattern is used for example. Therefore, not even prior work acknolwedges the need for naturalness-looking for auditing purposes.
	- 8. The method involves repeated model retraining with unrolled optimization, rematerialization, and truncated backpropagation—all computationally demanding procedures. However, the paper does not report runtime, GPU memory usage, or total training cost. Without this information, it is difficult to assess whether the framework is truly scalable beyond small benchmark datasets.
		- To answer the reviewer comment about resource usage, we have compiled Section <XXX> in the appendix that reports the requested metrics.
		- TODO  ==@Arielle== Can you take care of this?
	- Minor comments:
	    1. The acronym ERM (Empirical Risk Minimization) is used without definition and should be introduced upon first mention.
	- #### Questions
	    1.  The anonymous code link provided in the submission does not open. Could you please share a working repository or include a zip file in the supplementary material to ensure full reproducibility?
		- ==@Mohammad==
	- 2. Could you provide a more rigorous justification for treating the logit difference (Equation 5) as a valid surrogate for the likelihood-ratio statistic? Specifically, under what assumptions does maximizing this surrogate guarantee improved membership distinguishability, and can any theoretical bound or consistency argument be established?
		- ==@Mohammad==
	- 3. The paper claims that unrolled optimization provides “exact gradients,” yet the use of truncated backpropagation and rematerialization implies an approximation. Could you quantify how this truncation affects the final canary detectability? For instance, how does TPR@FPR vary as the truncation window K changes?
		- ==@Arielle==
	- 4. What measures were taken to ensure optimization stability across seeds and models? Do different initialization points (e.g., influence-selected seeds vs. random) lead to consistent canary detectability, or is the outcome highly variable?
		- ==@Arielle==, ==@Mohammad==
	- 5. Since the framework is inspired by differential privacy but ultimately empirical, can you clarify how the metric (TPR@FPR) relates to formal ε or δ values? Is there any attempt to estimate lower bounds on ε or compare to DP auditing baselines that produce numeric privacy budgets?
		- ==@Mohammad==
		- Above should probably be enough
		- Lower bounds requires many many attacks to establish meaningful bounds and even those are loose
	- 6. The optimization directly targets the LiRA hinge-based statistic. Would the optimized canaries remain highly detectable under alternative membership inference attacks (e.g., confidence-, entropy-, or loss-based)? Have you evaluated cross-auditor robustness?
		- The LiRA hinge loss (although poorly named) follows from likelihood tests with a prior assumption of normalcy (an assumption that given large sample size, the central limit theorem well supports).  Neyman-Pearson lemma establishes that thresholding this statistic is the optimal test. Given the principled, and optimal derivation of the prior work, we fail to see the need for using other test statistics that are more heuristic and much less adopted.
		- Can the reviewer kindly let us know, under what conditions it makes sense to use the aforementioned confidence-, entorpy-, or loss-based attacks; instead of LiRA?
		- ==@Nicolas @Florian== this one is pretty unreasonable to me. Does the above suffice?
	- 7. The transferability of optimized canaries across architectures is one of the key selling points of the paper. Can you provide a theoretical or empirical explanation for why canaries optimized on ResNet-9 remain highly detectable on ResNet-50 or WideResNet? Is this phenomenon architecture-dependent or data-dependent?
		- As discussed, a theoretical study of this phenomenon is outside of the scope of the current paper.
		- We provide the following observational explanation of this phenomenon:
			- All models (i.e. hypothesis classes) seek to learn the same concept  from the data.
			- A transferable canary indicates that the notion of a canary is not a function of the minutaie of the hypothesis class, but rather the concept itself.
			- For example, for digit classification, we know that a 2 and a 7 are reasonably close to each other; and one can be mistaken for the other. Therefore, a good canary can be an image that can reasonably be classified as either 2 or a 7 by even a human—and entirely different learner!
			- We like to note however that the space of canaries is potentially much larger than the above example. But the above should be sufficient to show why transferability makes sense in the first place.
	- 8.  Some optimized canaries, particularly in CIFAR-10, appear visually unnatural or off-manifold. Have you attempted to quantify the degree of deviation from the data distribution (e.g., via FID, nearest-neighbor distance, or classifier confidence)? Could detectability be driven by such distributional shifts rather than true memorization?
		- We have not. As discussed, our evaluation procedure is independent of our optimization procedure. In fact, there is no novelty on the evaluation front whatsoever.
		- Since out canaries fit the DP definition perfectly, we are not convinced that "visual unnaturalness" is a metric worth optimizing for. Afterall, the threat model is different from an attack using adversarial example so "bounding the (visual) perturbation" is not a requirement.
		- In the end, our guiding principle is the detectability using memebrship inference attacks; whatever means achieve this is considered fair game in privacy auditing (including changing the training procedure in substantial ways; see [[Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning]]). Our perturbed samples are far less invasive.
	- 9.  The paper emphasizes scalability, yet unrolled optimization and rematerialization are computationally heavy. Could you please report the runtime, GPU memory footprint, and training cost for the CIFAR-10 experiments? How would the approach scale to larger datasets or transformer architectures?
		- Question of runtime addressed above
		- Our final design for OptiFluence has several characteristics that simplifies scaling challenges. a) modularity; b) scalability knobs with truncation and influence calculation using EK-FAC approximations (which are scaled to transformers [[Studying Large Language Model Generalization with Influence Functions]]); c) first-party privacy auditing has a significant overhead. By showing transferability, one cost is amortized to multiple models; and even multiple parties.
- ### Reviewer_mvjG
  collapsed:: true
	- **Rating:** 8
	- **Confidence:** 3
	- #### Summary
	    The paper introduces OptiFluence, which the authors claim is a state of the art privacy canary generation method via a bilevel optimization program that maximizes the likelihood ratio for membership inference by first initializing the canary based on its influence on other samples and then performing an efficient (in compute and memory) gradient-based optimization to fine-tune these samples to maximize the likelihood ratio. They also introduce methods to make their canary optimization efficient in memory and compute by unrolling model updates and checkpointing the computational graph, or even truncating the gradient propagation.
	- #### Scores
		- Soundness: 4
		- Presentation: 4
		- Contribution: 3
	- #### Strengths
	    **[S1]** Very well-motivated and concretely described methodology, with an attention to detail to practical concerns like compute and memory costs, yielding a practical design that vastly outperforms baselines, which is outstanding.
	    
	    **[S2]** Use of influence functions to initialize canary is very well motivated and grounded in existing research, and its utility firmly corroborated by ablation studies.
	    
	    **[S3]** Speaking of which, all the components of the OptiFluence method are covered and ablated in the ablation study section, which very clearly shows each component’s significance. Put another way, I think this is a very well executed ablation study section. In addition, figure 3 provides a good overview of how all the components come together to yield strong canaries (high TPR at very low FPR) as compared to other (ablated) variants.
	    
	    **[S4]** Good takeaways, viz. pointing out the limitations of mislabeling for canary generation, illustrating how initializing with influence functions provides a more effective and principled approach.
	    
	    **[S5]** Transferability of generated canaries is a huge positive and contributes to efficient auditing practices.
	    
	    **[S6]** Auditing (with and without DP-SGD) is well done, with strong choices of auditing methods and well-executed DP-SGD training with a Renyi DP based accountant.
	    
	    **[S7]** The authors provide an anonymized link to the code and relevant hyperparameters in the appendix, aiding in reproducibility of their results.
	- #### Weaknesses
	    **[W1]** Not a serious weakness/dealbreaker, but it would be desirable to see results on more involved datasets than CIFAR-10 and MNIST. These datasets are popular classic datasets, so to speak, but it would be interesting to see if these results generalize to much larger datasets or datasets with many more classes than 10 (viz. CIFAR-100), more interesting sample distributions, or (this next part is not needed, so the authors can safely ignore this, but it would be appealing) other modalities than image datasets.
	    
	    **[W2]** Seeing as how attack success (unsurprisingly) degrades for low values of $\varepsilon$ for DP-SGD auditing, could the authors please add experiments on lower values of $\varepsilon$ (viz. 1 and <1)? While these values may yield lower utility of the model, theoretically they are desirable (especially <1) and it would be useful to see how Optifluence (and its baselines) perform in this regime.
	- #### Questions
	    **[Q1]** Can you address W1 and add results on more datasets in different regimes (more classes, samples, different distributions)?
	    
	    **[Q2]** Can you investigate the efficacy of Optifluence and its baselines in low $\varepsilon$ (high privacy) regimes for DP-SGD auditing?
	    
	    **[Q3]** Update: The other reviewers rightly point out the need to justify and corroborate the efficiency of unrolled updates in your paradigm with empirical evidence (runtime and memory used) and on more involved/expensive settings than CIFAR-10 and MNIST. Can the authors please address that? This is *key to me maintaining my current score*.
- ### Reviewer_5zJV
	- **Rating:** 4
	- **Confidence:** 4
	- #### Summary
	    This paper claims that existing privacy canaries, like mislabeled or out-of-distribution (OOD) points, are ad hoc and ineffective. It proposes to replace these guesses with a "principled" bilevel optimization framework. The method, OptiFluence, first finds a promising "seed" point from the real data using influence functions (IF-Init) and then uses computationally expensive unrolled optimization (ReMat+TBPTT) to fine-tune the sample's pixels for maximum detectability.
	- #### Scores
		- Soundness: 3
		- Presentation: 2
		- Contribution: 2
	- #### Strengths
	    1. The core idea is sound. Moving from "guessing" a canary to "optimizing" one is a logical step.
	    2. The results are undeniable. Table 1 shows that optimized canaries are 415x more detectable than standard ones on CIFAR-10.
	    3. The transferability result is the paper's strongest practical contribution.
	- #### Weaknesses
	    1. Calling this "scalable" in the title is a serious overstatement. The method is built on unrolled optimization, which is notoriously memory- and compute-intensive. The experiments are confined to MNIST and CIFAR-10. This will not scale to any model we actually care about auditing (e.g., LLMs).
	    2. Only report TPR @ Low FPR; other metrics such as AUC should also be considered.
	    3. The baseline of MIA is kind of outdated. There are more recent and powerful MIA attacks such as "Zarifzadeh, Sajjad, Philippe Liu, and Reza Shokri. "Low-cost high-power membership inference attacks." Proceedings of the 41st International Conference on Machine Learning. 2024."
	    4. The entire optimization objective is to maximize the LiRA "hinge" score. The canaries are overfit to this specific MIA. More MIA attacks should be considered.
	- #### Questions
	    1. The "scalable" claim is tested on CIFAR-10. What is the actual wall-clock time and VRAM cost?  How can you claim this is feasible for large-scale models?
	    2. For Table 2, what is the performance of DP-SGD?
	    3. Does your proposed method still work for other MIA?
- ### Reviewer_5uPC
	- **Rating:** 4
	- **Confidence:** 3
	- #### Summary
	    This work introduces OptiFluence, an optimization-based framework for conducting membership inference attacks (MIAs) against differentially private (DP) models. Unlike prior works in the central setting that often rely on sample-based canaries, OptiFluence formulates the attack as a gradient-based optimization problem in which the attacker crafts a synthetic sample to maximize their influence on the model output or loss. The authors argue that this exposes vulnerabilities in DP training not captured by existing stochastic or heuristic attacks. Experiments on standard benchmarks show that OptiFluence achieves higher attack success rates under certain scenarios.
	- #### Scores
		- Soundness: 2
		- Presentation: 3
		- Contribution: 2
	- #### Strengths
		- The optimization-based formulation is well described and the approach of combining influence functions to intitialize an optimized canary is novel.
		- The attack is evaluated across a good range of datasets and privacy budgets.
	- #### Weaknesses
		- The exact contribution over prior work that utilizes optimized canaries in the white-box or federated setting is unclear (see below).
		- Aspects of the presentation related to the specific names of proposed/baseline methods could be improved to make the experiments more easily readable (see questions below).
		- The optimization process seems to be computationally expensive. Although the authors propose an approximation denoted, ReMat+TBPTT, no experimental results demonstrate its effectiveness or runtime benefits. This makes it seem less practical than simpler alternatives (e.g., one-run or random canary methods).
			- No mention of threat models
			- What we propose is a method to optimize data (canary), one-run methods are auditing algorithms. They recieve canaries as an input; we produce canaries as output.
	- #### Questions
	    1. What exactly  is IF-OPT? It doesn’t seem to be clearly defined in the experiments.
	    2. Could the authors provide runtime/overhead comparisons against the baselines to substantiate claims of scalability? The approximation approach ReMat+TBPTT is proposed but there seem to be no results that show why you should use it over the fully unrolled updates. There are no statements in the paper about how long this canary optimization process takes? Is this attack really practical for auditing?
	    3. Given the higher computational cost, what practical advantage does OptiFluence offer over one-run or random canary insertion attacks?
	    4. I would have liked to have seen a clearer ablation on the impact of using the influence function intialization vs. standard canary optimization (w/o this IF initialization) which appears to be missing from Figure 3?
	    5. How does OptiFluence fundamentally differ from the optimization-based canaries proposed by [1,2]?
	    
	    [1] Nasr, Milad, et al. "Tight auditing of differentially private machine learning." 32nd USENIX Security Symposium (USENIX Security 23). 2023.
	    
	    [2] Maddock, Samuel, Alexandre Sablayrolles, and Pierre Stock. "Canife: Crafting canaries for empirical privacy measurement in federated learning." arXiv preprint arXiv:2210.02912 (2022).
- ### Reviewer_t7cL
	- **Rating:** 4
	- **Confidence:** 5
	- #### Summary
	    This paper proposes OptiFluence, a bilevel optimization framework for designing privacy canaries that maximize their detectability under membership inference attacks. The method integrates influence-based pre-selection and unrolled sample optimization with memory-efficient techniques. Experiments on MNIST and CIFAR-10 demonstrate strong detectability.
	- #### Scores
		- Soundness: 2
		- Presentation: 3
		- Contribution: 2
	- #### Strengths
	    * The paper tackles an important problem in empirical privacy auditing, providing a more principled alternative to heuristic canary constructions.
	    
	    * The bilevel optimization formulation is elegant and connects privacy auditing with influence functions and gradient unrolling.
	- #### Weaknesses
	    * **Incomplete component description in the abstract.**
	    The abstract claims that OptiFluence consists of three components but describes only two. Please correct this inconsistency.
	    
	    * **Lack of discussion on the relationship to adversarial examples.**
	    As noted around L228–L229 and Algorithm 1, the optimization process resembles adversarial example generation.
	    The paper should explicitly discuss how the proposed canary differs from conventional adversarial samples, conceptually and in objective formulation, and clarify why existing adversarial methods cannot directly be used to generate canaries. Experimental comparisons between OptiFluence and adversarial samples are also needed.
	    
	    * **Missing threat-model specification.**
	    The current presentation lacks a clear statement of the auditor’s capability—whether auditing assumes black-box, gray-box, or white-box access to the model.
	    A formal threat model is essential to contextualize the results and interpret the claimed transferability.
	- #### Questions
	    * **Unclear explanation of transferability.**
	    While Section 6.2 claims strong cross-architecture transfer, the supporting evidence in Appendix C.4 (two MLPs on MNIST) is insufficient to substantiate general transferability.
	    The authors should explain why optimized canaries can transfer between architectures (e.g., shared feature space, loss geometry) and include more diverse models or quantitative analyses.
	    
	    * **Potential performance degradation.**
	    Algorithm 1 suggests that the canary is iteratively updated during model training. This may interfere with model convergence or degrade performance.
	    The paper should report whether incorporating such optimization affects model accuracy or training stability.
	    
	    * **Relation to privacy–robustness trade-off.**
	    Given that OptiFluence’s optimization resembles adversarial training, it would be valuable to evaluate or at least discuss the potential trade-off between privacy auditing effectiveness and robustness, as widely documented in the literature [1–3].
	    An experiment showing canary auditing after adversarial training would significantly strengthen the paper.
	    
	    [1] Zhang, Z., Zhang, L. Y., Zheng, X., Abbasi, B. H., & Hu, S. (2022). Evaluating membership inference through adversarial robustness. The Computer Journal, 65(11), 2969-2978.
	    [2] Lyu, L. et al. (2022). Privacy and robustness in federated learning: Attacks and defenses. IEEE TNNLS, 35(7), 8726-8746.
	    [3] He, F., Fu, S., Wang, B., & Tao, D. (2020). Robustness, privacy, and generalization of adversarial training. arXiv preprint arXiv:2012.13573.