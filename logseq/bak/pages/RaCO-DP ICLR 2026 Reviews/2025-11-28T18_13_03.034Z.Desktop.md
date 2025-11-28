# Reviews for: Private Rate-Constrained Optimization with Applications to Fair Learning

- ### Reviewer_MESD
    - **Rating:** 8
    - **Confidence:** 3
    - #### Summary
        This work gives an algorithm for differentially private optimization in the setting of machine learning where we have "rate constraints," i.e., constraints that limit how the model predicts across different parts of the dataset. The main application is to fairness. As I understand it, no prior work considers this exact problem even non-privately, but we can compare with prior DP algorithms for important special instances. (The rate constraints here are adapted to multiclass classification and, unlike Cotter et al. (2019), use a differentiable notion of rate.)
        
        Algorithmically, we solve a Lagrangian formulation of this problem with a DP version of stochastic gradient descent-ascent: we privately update the primal parameters, then the dual parameters, then we project them back to the feasible space. Before this, for each step, we run a private histogram to estimate the current model's predictions on the current mini-batch.
        
        We get a theoretical convergence analysis (which seems to have to deal with several nontrivial obstacles) and experiments, where the proposed approach appears to essentially dominate prior work.
    - #### Scores
        - Soundness: 3
        - Presentation: 2
        - Contribution: 3
    - #### Strengths
        This seems like a very solid contribution. There's been a lot of work on DP optimization in both minmax and fairness settings, and this seems like it contains technical ideas that will be useful elsewhere. The algorithm will surely be an experimental baseline for future work.
    - #### Weaknesses
        I feel the presentation of the paper could be improved. Here are some areas I had particular difficulty:
        1. The rate constraints here adapt the Cotter et al. (2019b) definition by extending it to multiclass, but also working with soft decisions. The latter is a major point in their paper and may be an important distinction in the context of fairness [see 1].
        1. We also consider constraints that may look at overlapping parts of the dataset. I am still somewhat confused about this: am I given the rate constraints and then have to come up with the dataset partition myself? Does the manner in which I do this depend on the data?
        1. Section 3: the example partition is Hispanic, Black, or Caucasian. This is actively confusing, since often "Hispanic" is treated as an ethnicity distinct from a race, so for example on the US Census one can be "Hispanic Black" or "Non-Hispanic White."
        1. Section 4: what's the obstacle in obtaining the per-sample decomposition? What non-trivial operation is happening here?
        
        These were the main issues I ran into when reading the paper. They caused a fair bit of difficulty in forming my evaluation of it.
        
        [1] Canetti, Ran, et al. "From soft classifiers to hard decisions: How fair can we be?." Proceedings of the conference on fairness, accountability, and transparency. 2019.
    - #### Questions
        How are the rate constraints provided to Algorithm 1? Does the global partition depend on the rate constraints, or is it also supplied to the algorithm?

- ### Reviewer_AG8J
    - **Rating:** 6
    - **Confidence:** 3
    - #### Summary
        This paper proposes RaCO-DP, a DP variant of the stochastic gradient descent–ascent algorithm for solving the Lagrangian formulation of rate-constrained problems. This paper proposes an algorithm that enforces fairness and differential privacy simultaneously. They formulate “fairness” as a rate-constrained problem in empirical risk minimization. However, previous formulations apply only to binary classification tasks.
        
        This paper proposes generalized rate constraints that can handle multi-class classification. For privately solving the ERM with generalized rate constraints, they propose RaCO-DP, a DP variant of the stochastic gradient descent–ascent algorithm. Specifically, they leverage the fact that the rate-constraint objective can be expressed as a function of a histogram of model predictions, and that this histogram is easy to privatize (its sensitivity is 1) with the Laplace mechanism. RaCO-DP first computes a private histogram of the predictions, and then combines it with DP-SGD–style updates in the descent and ascent steps.
        
        Under standard assumptions in non-convex optimization, they prove that their algorithm achieves convergence to an approximate stationary point with a rate comparable to those in related minimization settings. They also provide experimental results showing that their algorithm achieves better fairness–accuracy trade-offs than previous algorithms across different privacy levels.
    - #### Scores
        - Soundness: 3
        - Presentation: 3
        - Contribution: 3
    - #### Strengths
        * The paper is clearly written and easy to read.
        * It has significance as it generalizes prior fairness constraints that previously limited to the binary classification setting to the multiclass setting. It also proposes a novel DP algorithm that leverages a private histogram; empirically, the method achieves a better privacy–fairness trade-off, and the paper also provides a theoretical convergence analysis for the algorithm.
        * It provides experimental investigations and ablations over different hyperparameter choices and privacy budgets. It also shows that it is computationally more efficient than prior work, with only ~2× per-step overhead compared with DP-SGD (and large speedups over DP-FERMI).
    - #### Weaknesses
        The method’s performance can degrade substantially when the smallest subgroup is tiny, because DP noise makes its histogram estimates inaccurate. The paper offers only limited discussion of this case.
    - #### Questions
        The convergence guarantee Theorem 5.2 does not depend on the rate constraint parameters. What are the dependencies?
        
        How valid is the bounded dual space assumption (section 5)? Since a compact $\Lambda$ caps penalties, it can induce violations to the rate constraints.

- ### Reviewer_vscK
    - **Rating:** 6
    - **Confidence:** 2
    - #### Summary
        RaCO-DP is a novel framework for optimizing machine learning models under rate constraints with differential privacy, DP variant of SGDA.
    - #### Scores
        - Soundness: 3
        - Presentation: 2
        - Contribution: 4
    - #### Strengths
        - Formal convergence analysis of RaCO-DP,
        - New SOTA claimed to be achieved on standard benchmarks for tabular datasets: CelebA, Parkinsons, ACSEmployment,
        - Scales to multiple sensitive groups (tested up to 18),
        - Scales beyond convex models to deep learning model (ResNet16 on CelebA),
        - Stronger privacy guarantees than former approaches,
        - Allows to specify directly maximum disparity.
    - #### Weaknesses
        - Results section is not clear, which exactly datasets displayed results achieving SOTA?
        - Authors didn't run the experiments for SOTA but compared with published data from Lowy et al.
    - #### Questions
        Improve presentation of results adding details regarding which datasets RaCO-DP exactly achieved a new SOTA on.

- ### Reviewer_sEj5
    - **Rating:** 6
    - **Confidence:** 3
    - #### Summary
        This paper tackles the challenge of training machine learning models under rate-based constraints (e.g. group fairness metrics like demographic parity, equalized odds) while ensuring differential privacy (DP). The authors propose RaCO-DP, a differentially private stochastic gradient descent-ascent method for solving the Lagrangian of constrained optimization problems that involve group-level rate constraints. A key innovation is the use of generalized rate constraints, which unify various constraint types as computations on subgroup histograms. By privately estimating these subgroup statistics at each training step (with calibrated noise), the method enforces constraints with minimal extra privacy cost. The paper provides a rigorous theoretical analysis, proving that RaCO-DP converges to an approximate stationary point even for non-convex problems. Empirically, the approach is evaluated on multiple fairness-sensitive benchmarks (enforcing demographic parity, false negative rate limits, and equalized odds) across tabular datasets (Adult, Credit, Parkinsons, ACSEmployment) and a deep learning task (CelebA with ResNet16). Results show that RaCO-DP achieves state-of-the-art trade-offs between accuracy, fairness, and privacy – Pareto-dominating prior private fairness methods under the same privacy budgets. Notably, the method scales to scenarios with many subgroups (18 in one experiment) and maintains high utility even under stringent privacy (e.g. $\varepsilon=1$) on a deep neural network. In summary, this work introduces the first general DP framework for rate-constrained learning, demonstrating strong theoretical guarantees and improved privacy–utility–fairness performance in fair ML applications.
    - #### Scores
        - Soundness: 3
        - Presentation: 3
        - Contribution: 3
    - #### Strengths
        - Novelty and Generality: The paper bridges a significant gap by presenting the first general DP optimization framework for arbitrary rate constraints, beyond prior work that focused almost exclusively on fairness constraints. This general formulation means the method could apply to a wide range of constrained ML tasks (fairness, robustness, cost-sensitive learning) under DP, marking a clear advancement in differentially private optimization.
        
         - Algorithmic Innovation: The proposed RaCO-DP algorithm (DP-SGDA) cleverly overcomes the core challenge that rate-based constraints depend on global group statistics (violating per-sample DP assumptions). The solution – privately computing subgroup histograms each mini-batch – ensures constraints can be evaluated and optimized without leaking individual data, at a privacy cost equivalent to a histogram query per step. This design exploits the linear structure of the dual (Lagrange multiplier) updates to contain noise growth. It’s an elegant extension of DP-SGD to constrained problems, and the authors describe it clearly with a unified “generalized rate constraint” formulation and examples.
        
         - Theoretical Rigor: The paper provides a solid theoretical foundation. The authors prove convergence of RaCO-DP to an approximate stationary point even for non-convex objectives. The analysis is non-trivial – it accounts for the bias introduced by noisy gradient estimates and leverages the linear structure of the constraints to improve convergence speed. Such theoretical guarantees (with formal proofs in the appendix) lend credibility to the approach and are a notable strength, as prior DP-fairness works often lacked convergence analysis.
        
         - Empirical Performance: The method shows excellent empirical results on multiple benchmarks. Across standard tabular datasets, RaCO-DP consistently achieves higher accuracy for the same fairness level (or lower disparity for the same accuracy) compared to prior approaches like DP-FERMI. In other words, it Pareto-dominates the previous state-of-the-art in the accuracy–fairness–privacy trade-off. For example, on a deep learning task (CelebA, ResNet16), RaCO-DP reaches ~90% accuracy with only a 10% demographic parity gap under strong privacy ($\varepsilon=1$), which is very close to the non-private model’s 95% accuracy. These are impressive results, demonstrating that the privacy cost of fairness constraints can be kept low.
    - #### Weaknesses
        - Soft Constraint Enforcement: RaCO-DP uses a Lagrangian (dual ascent) approach, effectively treating constraints in a soft manner during training. While this is standard in constrained optimization, it means the model might violate constraints transiently or converge to a point that satisfies them only approximately (within the allowed slack $\gamma$). The authors themselves discuss that using “hard” constraint enforcement (always projecting onto the constraint set) would break their theoretical guarantees, and empirically gave only limited gains. This is a minor weakness in the sense that the final solutions do seem to meet the constraints to the desired degree, but strictly enforcing constraints at all times isn’t supported by the core algorithm. In scenarios where absolutely no constraint violation can be tolerated, additional measures might be needed (at the cost of privacy or convergence guarantees).
        
         - Hyperparameter Sensitivity (Clipping): The approach inherits the need for per-sample gradient clipping from DP-SGD, which can introduce bias. In constrained optimization, an improperly tuned clipping norm can even prevent satisfying the constraints. The authors note that clipping can push the solution outside the feasible set, making the choice of clip norm “a critical hyperparameter”. In practice, this likely requires careful tuning or expert knowledge to get right – a potential drawback for ease of use. The paper does illustrate this issue (showing that overly small clip norms fail a strict FNR=0 constraint) and acknowledges it as a general DP limitation, but it remains a practical challenge.
        
         - Scope of Experiments: While the method is formulated generally, the empirical evaluation is focused primarily on group fairness scenarios. All reported experiments enforce fairness-related constraints; other applications (like robust optimization or cost-sensitive learning mentioned in the motivation) are not demonstrated. This leaves a gap in showing the framework’s generality in practice. For example, it would strengthen the paper to see at least one non-fairness use-case or a discussion of how the method would handle a different type of rate constraint. As is, the results convincingly cover fairness, but the general claim is not fully backed by experiments.
    - #### Questions
        1. RaCO-DP enforces constraints up to a slack $\gamma$. In practice, if a user needs to guarantee a strict constraint ($\gamma$ very close to 0), what is the advised approach? The discussion in the paper suggests using hard constraints in the dual update breaks theory and gives limited benefit. Could the authors clarify the observed impact of using hard constraints: did it ever significantly improve fairness or was it truly negligible? Also, in scenarios like requiring zero disparity or error ($\gamma=0$), does RaCO-DP struggle due to noise? It would help to know if there’s an inherent limit to how tight a constraint can be made under DP noise before performance collapses (as hinted by the FNR=0 case).
        
        2. The paper notes that the gradient clipping norm $C$ is critical and can affect the feasibility of constraints. How did you select the clipping norm in experiments, and how sensitive are the results to this choice? Is there a heuristic or adaptive strategy you can recommend for choosing $C$ to balance privacy (noise) and constraint satisfaction? Understanding this would help practitioners avoid the issue of failing to meet a constraint due to overly aggressive clipping.
        
        3. The method privately computes a histogram of subgroup counts per batch. How does the privacy budget consumption scale with the number of groups or constraints? For instance, in the ACSEmployment case with 18 subgroups, was there a noticeable increase in noise or decrease in accuracy relative to cases with fewer groups? Some discussion on scaling to a larger number of constraints (or larger partitions $Q$) under DP would be insightful. For example, does each additional subgroup simply incur a small constant cost, or do variance and utility trade-offs worsen significantly?
