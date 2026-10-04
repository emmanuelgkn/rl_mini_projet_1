import gymnasium as gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# =====================================================================
# 1. CALLBACK POUR MESURER LE BIAIS (Fonctionne pour DDPG et TD3)
# =====================================================================
class OverestimationBiasCallback(BaseCallback):
    """Mesure l'optimisme du Critique (Valeur Q sur l'état initial) sur TensorBoard."""
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        if self.num_timesteps % 2000 == 0:
            with th.no_grad():
                obs, _ = self.eval_env.reset()
                obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(self.model.device)
                
                # Action choisie par l'Acteur
                actions = self.model.policy.actor(obs_tensor)
                
                # Note estimée par le(s) Critique(s)
                q_values = self.model.policy.critic(obs_tensor, actions)
                
                # DDPG a 1 critique, TD3 en a 2. On prend le minimum pour généraliser.
                if isinstance(q_values, tuple) or isinstance(q_values, list):
                    estimated_q = th.min(th.cat(q_values)).item()
                else:
                    estimated_q = q_values.item()
                
            # Enregistrement dans TensorBoard
            self.logger.record("bias/estimated_q_initial", estimated_q)
        return True

# =====================================================================
# 2. FONCTION D'ENTRAÎNEMENT GÉNÉRIQUE
# =====================================================================
def run_experiment(algo_class, algo_name, hyperparams=None):
    print(f"\n🚀 Lancement de l'entraînement : {algo_name}...")
    
    env = Monitor(gym.make("LunarLanderContinuous-v3"))
    eval_env = gym.make("LunarLanderContinuous-v3")
    
    eval_callback = EvalCallback(
        Monitor(eval_env), 
        best_model_save_path=f"./logs/{algo_name}_best",
        log_path=f"./logs/{algo_name}_results", 
        eval_freq=2000, 
        deterministic=True, 
        render=False
    )
    bias_callback = OverestimationBiasCallback(eval_env)
    
    # Architecture réseau par défaut (2 couches cachées de 400 et 300 neurones)
    if hyperparams is None:
        hyperparams = dict(policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300])))

    # Initialisation du modèle (DDPG ou TD3)
    model = algo_class("MlpPolicy", env, tensorboard_log="./logs/tensorboard/", **hyperparams)
    
    # Entraînement (30 000 étapes suffisent pour voir les premières tendances)
    model.learn(total_timesteps=30_000, callback=[eval_callback, bias_callback])
    
    return f"./logs/{algo_name}_results/evaluations.npz"

# =====================================================================
# 3. GÉNÉRATION DES COURBES DE COMPARAISON
# =====================================================================
def plot_comparisons(results_dict):
    plt.figure(figsize=(10, 6))
    
    for label, npz_path in results_dict.items():
        data = np.load(npz_path)
        timesteps = data['timesteps']
        mean_rewards = np.mean(data['results'], axis=1)
        std_rewards = np.std(data['results'], axis=1)
        
        line, = plt.plot(timesteps, mean_rewards, label=label)
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color=line.get_color())
        
    plt.xlabel("Étapes d'entraînement")
    plt.ylabel("Score (Récompense cumulée)")
    plt.title("Comparaison des performances sur LunarLander")
    plt.legend()
    plt.grid(True)
    plt.show()

# =====================================================================
# 4. LANCEMENT DU PROJET
# =====================================================================
if __name__ == "__main__":
    results = {}
    
    # 1. Entraîner DDPG
    results["DDPG (Classique)"] = run_experiment(DDPG, "DDPG")
    
    # 2. Entraîner TD3
    results["TD3 (Classique)"] = run_experiment(TD3, "TD3")
    
    # 3. (Optionnel) Entraîner DDPG avec 3 couches cachées pour tester les hyperparamètres
    hyperparams_3_couches = dict(policy_kwargs=dict(net_arch=dict(pi=[400, 300, 200], qf=[400, 300, 200])))
    results["DDPG (3 couches)"] = run_experiment(DDPG, "DDPG_3_couches", hyperparams_3_couches)

    # Afficher toutes les courbes d'apprentissage sur le même graphique
    plot_comparisons(results)
