from main_e import *
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_overestimated_bias(**evaluators: Evaluator):                                        
    """                                                                                      
    Lit les logs TensorBoard des évaluateurs pour tracer le biais de surestimation.          
    """                                                                                         
    plt.figure()                           
    all_values = {}                                                  
    for name, evaluator in evaluators.items():                                               
        if evaluator.run_dir is None:                                                        
            print(f"L'évaluateur {name} n'a pas de répertoire de log (run_dir).")            
            continue                                                                         
                                                                                                
        # Chargement des événements enregistrés par Tensorboard                              
        event_acc = EventAccumulator(str(evaluator.run_dir))                                 
        event_acc.Reload()                                                                   
                                                                                                
        scalar_name = "bias/overestimation"                                                  
        
        # Vérifie si le biais a bien été loggué pour cet évaluateur
        if scalar_name in event_acc.Tags().get('scalars', []):
            events = event_acc.Scalars(scalar_name)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            all_values[name] = values
            plt.plot(steps, values, label=name)
        else:
            print(f"Aucune donnée '{scalar_name}' trouvée pour {name}")

    # Test de Welch pour DDPG (avec vs sans LayerNorm)
    if "DDPG_LN" in all_values and "DDPG_noLN" in all_values:
        val_ddpg_ln = all_values["DDPG_LN"][:-100]
        val_ddpg_noln = all_values["DDPG_noLN"][:-100]
        t_stat_ddpg, p_value_ddpg = stats.ttest_ind(val_ddpg_ln, val_ddpg_noln, equal_var=False)
        print(f"DDPG (LN vs noLN) overestimation: t={t_stat_ddpg:.2f}, p-value={p_value_ddpg:.5e}")

    # Test de Welch pour TD3 (avec vs sans LayerNorm)
    if "TD3_LN" in all_values and "TD3_noLN" in all_values:
        val_td3_ln = all_values["TD3_LN"][:-100]
        val_td3_noln = all_values["TD3_noLN"][:-100]
        t_stat_td3, p_value_td3 = stats.ttest_ind(val_td3_ln, val_td3_noln, equal_var=False)
        print(f"TD3 (LN vs noLN) overestimation: t={t_stat_td3:.2f}, p-value={p_value_td3:.5e}")


    plt.xlabel("etapes")
    plt.ylabel("biais de surestimation")
    plt.title("Évolution du biais de surestimation")
    plt.legend()
    output_path = Path("./images/overestimation.png")
    output_path.unlink(missing_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close()



def plot_reward(**evaluators: Evaluator):                                        
    """                                                                                      
    Lit les logs TensorBoard des évaluateurs pour tracer la reward.          
    """                                                                                         
    plt.figure()                                                                             
    for name, evaluator in evaluators.items():                                               
        if evaluator.run_dir is None:                                                        
            print(f"L'évaluateur {name} n'a pas de répertoire de log (run_dir).")            
            continue                                                                         
                                                                                                
        # Chargement des événements enregistrés par Tensorboard                              
        event_acc = EventAccumulator(str(evaluator.run_dir))                                 
        event_acc.Reload()                                                                                                                            
        scalar_name = "eval/reward"                                                  
        
        # Vérifie si le biais a bien été loggué pour cet évaluateur
        if scalar_name in event_acc.Tags().get('scalars', []):
            events = event_acc.Scalars(scalar_name)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            plt.plot(steps, values, label=name)
        else:
            print(f"Aucune donnée '{scalar_name}' trouvée pour {name}")

    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.title("Evolution de la reward")
    plt.legend()
    output_path = Path("./images/reward.png")
    output_path.unlink(missing_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close()
