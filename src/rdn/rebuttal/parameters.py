"""Maybe overkill but let's try to use pydantic The model validation I
blatantly implemented looking at claude, so I don't know if this is the best
way.
"""

import tomllib
from pydantic import BaseModel, model_validator
from jax import Array


class Unphosphorylated(BaseModel):
    diffusion_coeff: float
    k_in: float
    k_out: float
    initial_concentration: float


class Phosphorylated(BaseModel):
    diffusion_coeff: float


class Kinases(BaseModel):
    diffusion_coeff: float
    k_in: float
    k_out: float
    tau: float
    delta_stim: int
    sigma: float
    k_cat: float
    log_mu: float
    log_var: float
    log_cross_cov: float


class Phosphatases(BaseModel):
    diffusion_coeff: float
    k_in: float
    k_out: float
    tau: float
    delta_stim: int
    sigma: float
    k_cat: float
    log_mu: float
    log_var: float
    log_cross_cov: float


class Species(BaseModel):
    phosphorylated: Phosphorylated
    unphosphorylated: Unphosphorylated
    kinases: Kinases
    phosphatases: Phosphatases

    @model_validator(mode="after")
    def _warn_different_crosscorr(self):
        if self.kinases.log_cross_cov != self.phosphatases.log_cross_cov:
            print(
                f"\033[33m"
                "[WARNING] Different cross-correlation values defined in kinases"
                f" ({self.kinases.log_cross_cov})"
                " and phosphatases"
                f" ({self.phosphatases.log_cross_cov})"
                ". Using the kinases one."
                "\033[0m"
            )

        return self

    def __getitem__(self, name: str):
        return self.__dict__[name]

    def items(self):
        return self.__dict__.items()


class Dendrite(BaseModel):
    n_spines: int
    alpha: float


class Integration(BaseModel):
    model_type: str
    seed: int
    dt: float
    dx: float
    @model_validator(mode="after")
    def _check_validity_model_type(self):
        match self.model_type:
            case 'qss': pass
            case 'differential': pass
            case _: raise ValueError(
                'Invalid model type. Available choices are "differential"'
                'and "qss".'
            )
                
        return self


class Observation(BaseModel):
    spine_limits: list
    time_range: list


class UncagingProtocol(BaseModel):
    active: bool
    frequency: float
    duration: float
    spine_locations: list
    with_failure: bool


class Experiment(BaseModel):
    '''Contains the experimental decisions not regarding biology'''
    n_sessions: int
    time_unit: str
    observation: Observation
    uncaging_protocol: UncagingProtocol

    @staticmethod
    def time_unit_to_factor(unit: str):
        match unit:
            case "s":
                factor = 1
            case "min":
                factor = 60
            case "ms":
                factor = 1e-3
            case _:
                raise ValueError("Allowed units: 'min', 's', 'ms'")
        return factor


class Parameters(BaseModel):
    species: Species
    dendrite: Dendrite
    integration: Integration
    experiment: Experiment

    @model_validator(mode="after")
    def _unfinished_stimulation(self):
        experiment = self.experiment
        time_unit = experiment.time_unit
        last_observation = max(experiment.observation.time_range)
        last_stim = experiment.uncaging_protocol.duration

        if last_observation < last_stim:
            print(
                f"\033[33m"
                f"[WARNING] Final observation point"
                f" ({last_observation} {time_unit})"
                f" happens before the uncaging protocol has ended"
                f" ({last_stim} {time_unit}). Be aware of this!"
                "\033[0m"
            )

        return self

    @model_validator(mode="after")
    def _observation_out_of_boundaries(self):
        observation = self.experiment.observation
        dendrite = self.dendrite
        if observation.spine_limits[1] > dendrite.n_spines:
            print(
                f"\033[33m"
                f"[WARNING] Upper observation limit"
                f" ({observation.spine_limits[1]})"
                f" is higher than spine number"
                f" ({dendrite.n_spines})."
                f" Enforcing the latter as upper observation limit."
                "\033[0m"
            )
            self.experiment.observation.spine_limits[1] = dendrite.n_spines

        return self


    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            toml = tomllib.load(f)

        return Parameters(**toml)  # .model_validate(toml)
