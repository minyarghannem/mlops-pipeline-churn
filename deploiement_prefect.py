from pipeline_prefect import all_flow, train_flow, evaluate_flow

if __name__ == "__main__":
    print("Déploiement des flows en cours...")
    
    # Planifier le flow 'all' tous les jours à 2h du matin (Cron: minute heure jour mois jour_semaine)
    all_flow.serve(name="ml-pipeline-all", cron="0 2 * * *")
    
    # Vous pouvez aussi servir les autres
    # train_flow.serve(name="ml-pipeline-train")
    # evaluate_flow.serve(name="ml-pipeline-evaluate")
