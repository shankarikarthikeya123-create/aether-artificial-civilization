from dataclasses import dataclass


@dataclass
class BayesianHypothesis:
    name: str
    prior: float


class BayesianReasoner:
    """
    Lightweight Bayesian reasoning engine for AETHER.

    Uses:
        posterior ∝ prior × likelihood

    The final probabilities are normalized so that
    all hypotheses add up to 1.0.
    """

    def __init__(self):
        self.hypotheses: dict[str, BayesianHypothesis] = {}

    def add_hypothesis(self, name: str, prior: float):
        prior = max(0.0, min(1.0, float(prior)))

        self.hypotheses[name] = BayesianHypothesis(
            name=name,
            prior=prior,
        )

    def update(self, likelihoods: dict[str, float]):
        weighted = {}

        for name, hypothesis in self.hypotheses.items():
            likelihood = max(
                0.0,
                min(1.0, float(likelihoods.get(name, 0.0))),
            )

            weighted[name] = (
                hypothesis.prior * likelihood
            )

        total = sum(weighted.values())

        if total <= 0:
            if not weighted:
                return {}

            equal_probability = 1.0 / len(weighted)

            return {
                name: equal_probability
                for name in weighted
            }

        return {
            name: value / total
            for name, value in weighted.items()
        }

    def most_likely(self, likelihoods: dict[str, float]):
        posterior = self.update(likelihoods)

        if not posterior:
            return None

        return max(
            posterior,
            key=posterior.get,
        )

    def explain(self, likelihoods: dict[str, float]):
        posterior = self.update(likelihoods)

        return {
            "hypotheses": posterior,
            "most_likely": (
                max(posterior, key=posterior.get)
                if posterior
                else None
            ),
        }


# ============================================================
# AETHER DEFAULT BAYESIAN MODEL
# ============================================================

bayesian_reasoner = BayesianReasoner()

bayesian_reasoner.add_hypothesis(
    "resource_shortage",
    0.4,
)

bayesian_reasoner.add_hypothesis(
    "normal_consumption",
    0.6,
)