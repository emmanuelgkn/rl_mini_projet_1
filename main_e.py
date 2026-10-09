import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import rl_mind.envs
import numpy as np
from dataclasses import dataclass, replace
from scipy import stats
from scipy.stats import pearsonr
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path
from rl_mind.core import Action, Actor
from rl_mind.nn import build_mlp, soft_update
from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.collectors import TransitionCollector
from rl_mind.evaluation import Evaluator
from rl_mind.notebook import run_directory, setup_tensorboard, silence_known_warnings


def make_mlp(sizes: list[int], output_activation=None, use_layer_norm: bool = False) -> nn.Module:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            if use_layer_norm:
                layers.append(nn.LayerNorm(sizes[i+1]))
            layers.append(nn.ReLU())
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ContinuousQNetwork(nn.Module):
    """The Q-network $Q(s, a)$ for continuous actions"""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int, use_layer_norm: bool = False):
        super().__init__()
        self.model = make_mlp([obs_dim + action_dim, *hidden, 1], use_layer_norm=use_layer_norm)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Compute $Q(s, a)$ for a batch: `[B, obs_dim] x [B, action_dim] -> [B]`"""
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


class ContinuousDeterministicActor(Actor[Action]):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int, use_layer_norm: bool = False):
        super().__init__()
        self.model = make_mlp(
            [obs_dim, *hidden, action_dim], output_activation=nn.Tanh(), use_layer_norm=use_layer_norm
        )

    def forward(self, obs: Tensor) -> Action:
        return Action(value=self.model(obs))


class GaussianNoise(Actor[Action]):
    """Adds Gaussian noise to the actions of another actor (at training time)"""

    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs: Tensor) -> Action:
        action = self.actor(obs).value
        return Action(value=action + self.sigma * torch.randn_like(action))

    def act(self, obs: Tensor) -> Tensor:
        return self.actor.act(obs)


class Evaluator_with_overbias(Evaluator):
    """
    Construit un nouvelle evaluateur qui va stocker le résultat du calcul de 
    l'erreur de surestimation à chaque eval_interval pas.
    
    """

    def __init__(self,
        env: VecEnv,
        every: int,
        run_dir: Path | None = None,
        writer: SummaryWriter | None = None,
    ):
        super().__init__(env, every,run_dir,writer)
        self.overbias_error = []

def compute_critic_loss(
    gamma: float, batch: Transitions[Action], q_values: Tensor, next_q_values: Tensor
) -> Tensor:
    """Compute the DDPG critic loss from a batch of transitions

    :param gamma: The discount factor
    :param batch: The batch of transitions
    :param q_values: $Q(s_t, a_t)$ from the critic (shape `[B]`)
    :param next_q_values: $Q'(s_{t+1}, \\pi(s_{t+1}))$ from the target critic
        (shape `[B]`)
    :return: The critic loss (a scalar)
    """
    # Compute the target (do not bootstrap when `batch.terminated`), then the MSE loss
    
    target = batch.reward + gamma * next_q_values * (~batch.terminated)

    return F.mse_loss(q_values, target)

def compute_actor_loss(q_values: Tensor) -> Tensor:
    """Return the actor loss given $Q(s_t, \\pi(s_t))$ (shape `[B]`)"""
    # Compute the actor loss

    return -torch.mean(q_values)

@dataclass(frozen=True)
class DDPGConfig:
    env_name: str = "LunarLanderContinuous-v3"
    seed: int = 1

    #: Total number of environment steps
    max_steps: int = 30_000
    #: Number of parallel training environments
    n_envs: int = 1
    #: Environment steps between two gradient updates
    steps_per_update: int = 1
    #: Steps before learning starts
    learning_starts: int = 1_000

    #: Replay buffer capacity
    buffer_size: int = 200_000
    batch_size: int = 64

    #: Discount factor
    gamma: float = 0.98
    #: Target network update coefficient
    tau: float = 0.05
    #: Exploration noise
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64) # c'est ici pour changer le nombre de couches cachée
    use_layer_norm: bool = False
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3

    #: Steps between two evaluations, and number of evaluation episodes
    eval_interval: int = 200
    n_eval_envs: int = 10

@dataclass(frozen=True)
class DDPGConfigLN:
    env_name: str = "LunarLanderContinuous-v3"
    seed: int = 1

    #: Total number of environment steps
    max_steps: int = 30_000
    #: Number of parallel training environments
    n_envs: int = 1
    #: Environment steps between two gradient updates
    steps_per_update: int = 1
    #: Steps before learning starts
    learning_starts: int = 1_000

    #: Replay buffer capacity
    buffer_size: int = 200_000
    batch_size: int = 64

    #: Discount factor
    gamma: float = 0.98
    #: Target network update coefficient
    tau: float = 0.05
    #: Exploration noise
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64) # c'est ici pour changer le nombre de couches cachée
    use_layer_norm: bool = True
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3

    #: Steps between two evaluations, and number of evaluation episodes
    eval_interval: int = 200
    n_eval_envs: int = 10

def log_overestimation_bias(
    writer: SummaryWriter,
    step: int,
    critic: nn.Module,
    actor: Actor,
    eval_env: VecEnv,
    gamma: float,
    evaluator : Evaluator_with_overbias | None=None,
):
    """
    Calcule le biais de surestimation en comparant le Q estimé au vrai retour Monte Carlo
    escompté, calculés sur la même paire (état, action) initiale.
    """
    with torch.no_grad():
        obs = eval_env.reset() #On remet à jour les 10 environnements d'évaluation
        actions = actor(obs).value #On predit leurs actions avec l'acteur 
        estimated_q = critic(obs, actions).mean().item() #On estime Q avec le critique 
        step_result = eval_env.step(actions) #On met à jour l'état de chaque agent après avoir effectué leurs actions
        obs = step_result.obs #On récupère chaque nouveaux états 
        rewards = step_result.reward #On récupère les gains de la transition
        terminated = step_result.terminated 
        truncated = step_result.truncated
        dones = terminated | truncated # On verifie que tous les environnements ne sont pas terminés ou tronqués 
        
        mc_returns = rewards.clone() #On initialise le retour de Monte Carlo comme la table des gains que l'on va mettre à jours 
        discounts = torch.ones_like(rewards) * gamma #On multiplie par le facteur d’actualisation
        
        while not dones.all(): #Tant que les environnements n'ont pas été interrompus, on prédit les actions suivantes par l'acteur, on met à jour les environnements et on récupère leurs gains qu'on multiplie par le facteur d'actualisation exponentié au numéro d'étape qu'on ajoute à la table du retour 
            actions = actor(obs).value
            step_result = eval_env.step(actions)
            obs = step_result.obs
            rewards = step_result.reward
            term = step_result.terminated
            trunc = step_result.truncated
            mc_returns += discounts * rewards * (~dones)
            discounts *= gamma
            dones |= (term | trunc)
            
        true_return = mc_returns.mean().item() 
    
    bias = estimated_q - true_return #on déduit le biais en faisant la différence de l'estimation du gain par le critique et le retour de Monte Carlo
    
    writer.add_scalar("bias/overestimation", bias, step)
    writer.add_scalar("bias/estimated_q", estimated_q, step)
    writer.add_scalar("bias/true_return", true_return, step)
    if evaluator is not None :
        evaluator.overbias_error.append(bias) #On enregistre ce biais en l'ajoutant à l'attribut overbias_error de l'evaluateur.
        
    return bias

def run_ddpg(cfg: DDPGConfig) -> Evaluator:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    # The actor, the critic and its target
    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim, cfg.use_layer_norm
    )
    target_actor = copy.deepcopy(actor)
    critic = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.use_layer_norm)
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # Data collection (with exploration noise), replay buffer and evaluation
    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    run_dir = run_directory(f"ddpg-{cfg.env_name}-S{cfg.seed}")
    evaluator = Evaluator_with_overbias(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Update the critic

        # Q-values of the actions that were played (this is where gradients flow)
        q_values =  critic(batch.obs, batch.action.value)
        # Q-values of the *current* actor's actions in the next states,
        # estimated by the target critic (no gradient!)
        with torch.no_grad():
            next_actions = target_actor(batch.next_obs).value
            next_q_values = target_critic(batch.next_obs, next_actions)

        critic_loss = compute_critic_loss(cfg.gamma, batch, q_values, next_q_values)
        # assert False, 'Not implemented yet'


        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update the actor (maximize Q(s, pi(s)))
        current_actions = actor(batch.obs).value
        actor_loss = compute_actor_loss(critic(batch.obs, current_actions))


        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the target critic and actor
        soft_update(critic, target_critic, cfg.tau)
        soft_update(actor, target_actor, cfg.tau)

        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            bias=log_overestimation_bias(evaluator.writer, collector.steps, critic, actor, evaluator.env, cfg.gamma, evaluator)
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f} bias_error={bias:7.1f}"
            )

    pbar.close()
    return evaluator





class TD3Config(DDPGConfig):
    #: Number of critic updates between two policy updates
    policy_delay: int = 2
    #: Std of the noise added to the target policy actions
    target_noise: float = 0.2
    #: Clipping of the target policy noise
    target_noise_clip: float = 0.5

@dataclass(frozen=True)
class TD3ConfigLN(DDPGConfigLN):
    #: Number of critic updates between two policy updates
    policy_delay: int = 2
    #: Std of the noise added to the target policy actions
    target_noise: float = 0.2
    #: Clipping of the target policy noise
    target_noise_clip: float = 0.5

def run_td3(cfg: TD3Config) -> Evaluator:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    # Create the actor, the two critics, their targets and the optimizers
    actor = ContinuousDeterministicActor(env.observation_dim, cfg.actor_hidden, env.action_dim, cfg.use_layer_norm)
    target_actor = copy.deepcopy(actor)

    critic_1 = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.use_layer_norm)
    target_critic_1 = copy.deepcopy(critic_1)

    critic_2 = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.use_layer_norm)
    target_critic_2 = copy.deepcopy(critic_2)

    actor_optimizer = torch.optim.Adam(actor.parameters(),lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(
        list(critic_1.parameters()) + list(critic_2.parameters()), lr=cfg.lr_critic
    )
    # assert False, 'Not implemented yet'

    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    run_dir = run_directory(f"td3-{cfg.env_name}-S{cfg.seed}")
    evaluator = Evaluator_with_overbias(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )

    updates = 0
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Implement the TD3 update

        # 1. Critic update: compute the common target with the target actor
        # (+ clipped noise) and the min of the two target critics, then
        # update both critics.
        # 2. Every `cfg.policy_delay` updates: update the actor using
        # critic_1, and softly update the three target networks.
        updates += 1

        with torch.no_grad():
            next_actions = target_actor(batch.next_obs).value
            noise = torch.randn_like(next_actions)*cfg.target_noise
            noise = torch.clamp(noise,-cfg.target_noise_clip, cfg.target_noise_clip)
            next_actions = torch.clamp(next_actions+noise,-1,1)

            next_q1 = target_critic_1(batch.next_obs, next_actions)
            next_q2 = target_critic_2(batch.next_obs, next_actions)
            next_q = torch.min(next_q1,next_q2)

        q1_values = critic_1(batch.obs, batch.action.value)
        q2_values = critic_2(batch.obs, batch.action.value)

        critic1_loss = compute_critic_loss(cfg.gamma, batch, q1_values, next_q)
        critic2_loss = compute_critic_loss(cfg.gamma, batch, q2_values, next_q)
        critic_loss = critic1_loss + critic2_loss

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if updates % cfg.policy_delay == 0:                                                        
                current_actions = actor(batch.obs).value                                               
                actor_loss = compute_actor_loss(critic_1(batch.obs, current_actions))                  
                                                                                                       
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                soft_update(actor, target_actor, cfg.tau)
                soft_update(critic_1, target_critic_1, cfg.tau)
                soft_update(critic_2, target_critic_2, cfg.tau)
                evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)          
                
        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps) 


        if result := evaluator.run_if_needed(collector.steps, actor):
            bias = log_overestimation_bias(evaluator.writer, collector.steps, critic_1, actor, evaluator.env, cfg.gamma, evaluator)
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f} bias_error={bias:7.1f}"
            )

    pbar.close()
    return evaluator


