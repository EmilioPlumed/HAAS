from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Hypothesis:
    id: str
    # Injected verbatim as the top-level system-prompt commitment
    statement: str
    # Why this is heterodox — used in logging and paper context
    rationale: str


HYPOTHESIS_POOL: Dict[str, Hypothesis] = {
    h.id: h
    for h in [
        Hypothesis(
            id="no_skip",
            statement=(
                "Eliminating skip connections improves learned representations "
                "despite training instability. Architectures without residual or "
                "dense connections force the network to learn richer intermediate "
                "features rather than relying on identity shortcuts."
            ),
            rationale=(
                "Contradicts the dominant post-ResNet consensus that skip connections "
                "are essential for depth. If correct, removes a major structural "
                "constraint from architecture design."
            ),
        ),
        Hypothesis(
            id="no_attention",
            statement=(
                "Attention mechanisms are redundant when sufficiently deep feedforward "
                "layers are used. With enough depth, MLP-based token mixing captures "
                "the same dependencies that attention learns, without quadratic cost."
            ),
            rationale=(
                "Challenges the Transformer dominance narrative. Directly relevant to "
                "efficient architecture design and the theoretical justification for "
                "attention."
            ),
        ),
        Hypothesis(
            id="asymmetric_depth",
            statement=(
                "Asymmetric encoder/decoder depth ratios — specifically, a much deeper "
                "encoder paired with a shallow decoder — outperform symmetric "
                "architectures on sequence tasks. The encoder should do the heavy "
                "representational work; the decoder should be cheap."
            ),
            rationale=(
                "Symmetric architectures are the default without strong justification. "
                "Asymmetric depth allocation could unlock better parameter efficiency."
            ),
        ),
        Hypothesis(
            id="shared_weights",
            statement=(
                "Sharing weights across all layers of a network reduces overfitting "
                "more effectively than dropout, while also providing an implicit "
                "regularization that improves generalization on small datasets."
            ),
            rationale=(
                "Weight sharing collapses depth into an iterative refinement process. "
                "Heterodox because it conflicts with the standard view that layer "
                "specialization is essential."
            ),
        ),
        Hypothesis(
            id="pure_conv",
            statement=(
                "Purely convolutional token mixing outperforms attention for tasks "
                "involving sequences shorter than 512 tokens. Locality bias is an "
                "asset, not a limitation, when global context is not needed."
            ),
            rationale=(
                "Directly challenges the blanket adoption of attention for all sequence "
                "lengths. Relevant for efficient small-sequence model design."
            ),
        ),
        # --- NAS-Bench-201 compatible hypotheses ---
        Hypothesis(
            id="sparse_cell",
            statement=(
                "Sparse inter-node connectivity within architecture search cells — "
                "maximizing dead (none) edges and activating only a small number of "
                "operations — forces each active operation to develop more expressive "
                "representations. Peak performance comes from a few strong, specialized "
                "connections rather than dense connectivity. Sparsity is a structural "
                "prior that improves generalization, contrary to DenseNet-style wisdom "
                "that more connections are strictly better."
            ),
            rationale=(
                "Directly contradicts the dense-connectivity consensus established by "
                "DenseNet and propagated through NAS literature, where denser cells "
                "consistently outperform sparse ones. If correct, most NAS search "
                "budgets are wasted on unnecessary connections."
            ),
        ),
        Hypothesis(
            id="no_3x3",
            statement=(
                "3x3 convolutions are an unnecessary computational primitive within "
                "architecture search cells. Pointwise (1x1) transformations, stacked "
                "across sufficient depth, capture the same spatial relationships at "
                "lower parameter cost. The spatial locality bias of 3x3 kernels is a "
                "historical artifact of hand-designed networks, not a structural "
                "necessity — and excluding them forces the cell to learn more "
                "efficient, generalizable representations."
            ),
            rationale=(
                "3x3 convolution is the unquestioned workhorse of every competitive "
                "NAS cell. Challenging it directly tests whether the field has "
                "over-indexed on a single operation. If 1x1 convolutions are "
                "sufficient, current NAS search spaces are unnecessarily constrained."
            ),
        ),
    ]
}
