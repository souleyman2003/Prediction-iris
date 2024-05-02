#!/usr/bin/env python

# coding: utf-8

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Créer une application Streamlit
def main():
    # Afficher un titre
    st.title("Prédiction de l'espece de la fleur")

    # Chargement du modèle
    model = joblib.load('iris.pkl')

    # Obtenir les données du client
    Longueur_petal = st.number_input('Petal.Lenght', min_value=0, max_value=850, value=600)
    Largeur_petal = st.number_input('Petal.Width', min_value=0, max_value=850, value=600)
    Longueur_sepal = st.number_input('Sepal.Lenght', min_value=0, max_value=850, value=600)
    Largeur_sepal = st.number_input('Sepal.Width', min_value=0, max_value=850, value=600)
    
    

    

    # Préparer les données de la fleur
    flower_data = np.array([[Longueur_sepal, Largeur_sepal, Longueur_petal, Largeur_petal]])

    # Utiliser le modèle pour prédire la probabilité 
    prediction = model.predict(flower_data)

    # Afficher la prédiction
    # Display prediction
    if st.button("Predict Species"):
        st.write('L\'espèce de la fleur est :', prediction)


if __name__ == '__main__':
    main()