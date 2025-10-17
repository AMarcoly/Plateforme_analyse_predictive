# Utils.py

def print_model_scores(scores_dict):
    print("Scores des modèles :")
    print("{:<20} {:<10} {:<10} {:<10}".format("Modèle", "Accuracy", "Precision", "Recall"))
    print("-"*55)
    for model, metrics in scores_dict.items():
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            model, metrics['accuracy'], metrics['precision'], metrics['recall']
        ))
