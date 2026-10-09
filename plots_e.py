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



def plot_evaluations_performance_versus_overbias(only_bias=False,**evaluators: Evaluator):

    """
    Représente sur un même graphe, la variation de la performance 
    (le gain cumulé moyen obtenu sur les n_envs environnements simulés en parallèle, en suivant la politique de son acteur)
    de chaque algorithme avec son biais de surestimation en fonction du nombre de pas d'entrainement.
    """
    
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx() #On crée un nouvel axe sur la figure pour représenter également l'erreur de surestimation. 

    palette = ["blue","orange"] #On affecte une palette par défaut des performances de ddpg et td3
    palette_overbias = ["dodgerblue","darkorange"] # On affecte une palette par défaut de l'erreur de suréstimation de ddpg et de TD3

    
    for i,(name, evaluator) in enumerate(evaluators.items()):
        
        
        steps = [result.step for result in evaluator.history]
        means = torch.tensor([result.mean for result in evaluator.history])

        print(f"Nombre de valeurs de gains moyens de {name} : ",len(means))

        
        
        stds = torch.tensor(
            [float(result.rewards.std()) for result in evaluator.history]
        )

        
        (line,) = ax1.plot(steps, means, label=f"gain cumulé de {name} avec une variabilité de +/- écart type ",color=palette[i])
        ax1.fill_between(
                steps, means - stds, means + stds, alpha=0.2, color=line.get_color()
            )

        if not only_bias and hasattr(evaluator, 'overbias_error'): #On vérifie que l'on s'intéresse à l'erreur de surestimation et que l'evaluateur contient l'attribut overbias_error
            overbias = evaluator.overbias_error
            (line,) = ax2.plot(steps, overbias, label= f"biais de surestimation de {name}",c=palette_overbias[i]) #On trace les courbes d'erreur de surestimation sur l'autre axe


            print(f"Nombre de valeurs de biais de surrestimation de {name} : ",len(overbias), "\n") 



            
            
            
            correlation, p_value = pearsonr(means.numpy(), overbias) #On calcule le test de corrélation de Pearson entre les gains moyens et l'erreur de surestimation

            res = stats.spearmanr(means.numpy(), overbias) #On calcule le test de corrélation de Spearman (cas plus général) entre les gains moyens et l'erreur de surestimation

            correlation_spearman, pvalue_spearman = res.statistic,res.pvalue
            

            

            print("------------------------------------------------------------")
            print(f"Test de Correlation entre reward et overbias pour la variable {name} Pearson")
            print(f"Coefficient de corrélation : {correlation:.2f}")
            print(f"P-valeur : {p_value}")
            print("------------------------------------------------------------")
            print(f"Test de Correlation entre reward et overbias pour la variable {name} Spearman")
            print(f"Coefficient de corrélation : {correlation_spearman:.2f}")
            print(f"P-valeur : {pvalue_spearman}")
            print("------------------------------------------------------------\n")

        
        
            

    lines, labels = ax1.get_legend_handles_labels() #On récupère les légendes des tracées sur chaque axe
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, framealpha=0.1 ,loc=2) # On les concatene en une unique légende que l'on place en haut à gauche de la figure

    
    ax1.set_xlabel("étapes")
  
    ax2.set_ylabel("biais de surestimation")

    if not only_bias :

        ax1.set_ylabel("moyenne des gains cumulés")

        plt.title("Variation de la moyenne des gains (Performance) et du biais \n" + "de surestimation par rapport au numéro d'étape et aux algorithmes testés")

        plt.savefig("./images/cumulated_reward_and_overbias_error_with_respect_to_the_steps")

    else:
        plt.title("Variation des biais de surestimation par rapport au numéro d'étape et aux algorithmes testés")

        plt.savefig("./images/overbias_error_with_respect_to_the_steps")
        
    plt.show()