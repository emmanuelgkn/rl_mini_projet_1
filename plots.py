from main import *

def plot_overestimated_bias(**evaluators: Evaluator):                                        
    """                                                                                      
    Lit les logs TensorBoard des évaluateurs pour tracer le biais de surestimation.          
    """                                                                                      
    try:                                                                                     
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  
    except ImportError:                                                                      
        print("Veuillez installer tensorboard pour utiliser cette fonction (pip install tensorboard).")                                                                                
        return                                                                               
                                                                                                
    plt.figure()                                                                             
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
            
            plt.plot(steps, values, label=name)
        else:
            print(f"Aucune donnée '{scalar_name}' trouvée pour {name}")

    plt.xlabel("steps")
    plt.ylabel("overestimation bias")
    plt.title("Évolution du biais de surestimation")
    plt.legend()
    plt.show()
