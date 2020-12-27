# -*- coding: utf-8 -*-

"""
Created on Sat Jul 25 14:57:28 2020

@author: Reda-Acer

"""

import pandas as pd
import numpy as np
import math
import ta
import matplotlib.pyplot as plt
from sklearn import preprocessing
import keras
from keras import backend as K
import tensorflow as tf
tf.enable_eager_execution()


# Importation des données à un tableau "dt":
dt = pd.read_csv('C:\\Users\\Acer\\Desktop\\Data Projects\\Algo trading\\data\\Maroc_Telecom.csv')    

'''
Nettoyage des données
'''

# Nombre d'observations manquantes:
dt.isna().sum()

# Check s'il n'y a pas d'autres valeurs manquantes:
dt['Close'].isna().sum() + dt['Close'].notna().sum() == len(dt)


# Supprime la colonne "Adj Close" car on n'en a pas besoin:
dt = dt.drop('Adj Close', axis=1)


# Graphique du prix:
x = range(0, len(dt)) 
y = dt['Close']

fig, ax = plt.subplots()
ax.plot(x,y)
ax.set(xlabel='Jours', ylabel="Cours de l'action",
       title='Evolution du cours de Maroc Telecom')
ax.grid()
plt.show()

# Statistiques descriptives:
dt.describe()

# Auto-corrélation:
def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))
autocorr(df['Close'])


# Interpolation par splines d'Akima:
dt = dt.interpolate(method='akima')

# Si on veut exporter les données en format excel
# dt.to_excel("C:\\Users\\Acer\\Desktop\\Data Projects\\Algo trading\\donnees.xlsx")

'''
Attributs
'''

# Centile n-ème roulant:
dt['Prix Centile'] = ''
n = 10 # Prix centile à ordre 10
for i in range(n, len(dt)-n):
    liste_n = dt['Close'][i-n:i] # Donne la dernière valeur n pour chaque i-ème valeur qu'on input
    pr = dt['Close'][i] # prix de cloture
    compteur = 0
    for o in range(i-n, i):
        if pr > liste_n[o]:
            compteur += 1
        else:
            pass
    prix_centile = compteur/n
    dt['Prix Centile'][i+1] = prix_centile
    

# Remplit les vides:
dt['Percentile Price'][0:n+1] = 0.9
dt['Percentile Price'][len(dt)-n+1:len(dt)]=0.1


# Variable: Delta Prix

dt['delta_prix'] = ''
for i in range(1,len(dt)):
    dt['delta_prix'][i] = dt['Close'][i] - dt['Close'][i-1]
    
dt['delta_prix'][0] = 0


# Variable VWAP:

VWAP = ta.volume.VolumeWeightedAveragePrice(high = dt['High'], low = dt['Low'], close = dt['Close'], volume = dt['Volume'], n= 14, fillna = True)

# Ajoute la variable au tableau de données dt:
dt['VWAP'] = VWAP.volume_weighted_average_price()


# Variable: Bandes de Bollinger
bb = ta.volatility.BollingerBands(close=dt["Close"], n=20, ndev=1, fillna = True)

# Ajoute les attributs:
dt['bb_ma'] = bb.bollinger_mavg()
dt['bb_high'] = bb.bollinger_hband()
dt['bb_low'] = bb.bollinger_lband()

# Ajoute les deux variables catégoriques issues des bandes bollinger:
dt['bb_hi'] = bb.bollinger_hband_indicator()
dt['bb_li'] = bb.bollinger_lband_indicator()


# Ajoute une colonne où l'on calcule les rendements journaliers (pour les récompenses)
dt['rendement_journalier']= ''

for i in range(1, len(dt)):
    dt['rendement_journalier'][i] = float(( (dt['Close'][i]-dt['Close'][i-1] ) / dt['Close'][i-1])*100 )

dt['rendement_journalier'][0] = np.mean(dt['rendement_journalier'][1:4])
dt['rendement_journalier'] = pd.to_numeric(dt['rendement_journalier']) # transforme la nouvelle colonne au type "float"


# Histogramme des rendements:

mu = np.mean(df['rendement_journalier'])
sigma = np.std(df['rendement_journalier'])

num_bins = 70
fig, ax = plt.subplots()
n, bins, patches = ax.hist(df['rendement_journalier'], num_bins, density=1)

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * # Courbe théorique de la loi normale
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')

ax.set_xlabel('Rendements journaliers (%)')
ax.set_ylabel('Densité')
ax.set_title(r'Distribution des rendements journaliers Maroc Telecom')

plt.show()


#  Supprime les colonnes inutiles:
dt = dt.drop(['bb_ma', 'bb_high', 'bb_low','Date'], axis=1)

# ransforme le tableau final en dataframe:
df = pd.DataFrame(dt)

        
'''
Prétraitements:
'''

# Normalisation volume:
norm_volume = pd.DataFrame(preprocessing.normalize(np.array(dt['Volume']).reshape(1,-1)))
df['Volume'] = np.array(norm_volume).reshape(-1,1)

df['Volume'] = df['Volume']*10

# Création de l'environnement:

class Environment:
    
    def __init__(self):
        pass
        
    # Espace des actions:
    @staticmethod                 
    def espace_actions(action):   # Prend 0,1,2,3 ou 4 comme inputs et retourne l'ensemble des actions disponibles dans l'état suivant s' donnant l'état actuel s
        return [2,3] if action<=2 else [0,1,4]
        # 0 = Achat ; 1 = Vente ; 2 = Maintien ; 3 = Cloture ; 4 = Cash
    
    # à observer chaque étape (soit chaque jour):
    # Prend un indice "self" et retourne l'état actuel:
    def etat_actuel(self):
        return list(df.iloc[self, 0:len(df.columns)-1])
     
                                                         
   
    # Outputs next etat given current etat by taking current etat's index as self:
    def etat_suivant(self):
        return list(df.iloc[self + 1, 0:len(df.columns)-1])
            

    # On présente ici plusieurs fonctions de récompense candidates desquelles ont peut choisir (le choix dans le projet a été "recomp2")
    def recompense(self, indice_etat): # Fonction qui retourne la récompense pour chaque action qu'on lui donne
        
        if self == 4: # cash
            return -0.1  # récompense négative de rester en cash
            
        
        elif self == 0: # achat
            if df['rendement_journalier'][indice_etat+1] > 0:
                return df['rendement_journalier'][indice_etat+1] # récompense positive

            else:
                return -df['rendement_journalier'][indice_etat+1]  # pénalité
        
        
        
        elif self == 1: # vente
            if df['rendement_journalier'][indice_etat+1] < 0:
                return df['rendement_journalier'][indice_etat+1] # récompense positive
            else:
                return -df['rendement_journalier'][indice_etat+1] # pénalité
                
            
        elif self == 2: # Maintien:
            
            if derniere_action == 0: # maintien d'achat
                
                if df['rendement_journalier'][indice_etat] > 0:
                    return df['rendement_journalier'][indice_etat+1] # recompense positive
                else:
                    return -df['rendement_journalier'][indice_etat+1] # pénalité
                    
            elif derniere_action == 1: # maintien de vente
                if df['rendement_journalier'][indice_etat] < 0:
                    return df['rendement_journalier'][indice_etat+1] # recompense positive
                else:
                    return -df['rendement_journalier'][indice_etat+1] # pénalité
                
            elif derniere_action == 2: 
                
                for _ in range(len(ordres)-1, -1, -1): 
                    if ordres[_] == 0: 
                        if df['rendement_journalier'][indice_etat+1] > 0:
                            return df['rendement_journalier'][indice_etat+1] 
                        else:
                            return -df['rendement_journalier'][indice_etat+1]
                        break
                    
                    
                    elif ordres[_] == 1:
                        if df['rendement_journalier'][indice_etat+1] < 0:
                            return df['rendement_journalier'][indice_etat+1]
    
                        else:
                            return -df['rendement_journalier'][indice_etat+1]
                        break
            
            
        
        elif self == 3: # cloture:
            if ordres[len(ordres)-2] == 0:
                if df['rendement_journalier'][indice_etat+1] > 0:
                    return -df['rendement_journalier'][indice_etat+1] 
    
                else:
                    return df['rendement_journalier'][indice_etat+1]
            
            elif ordres[len(ordres)-2] == 1:
                if df['rendement_journalier'][indice_etat+1] < 0:
                    return -df['rendement_journalier'][indice_etat+1]
    
                else:
                    return df['rendement_journalier'][indice_etat+1]
            
            elif ordres[len(ordres)-2] == 2:
                
                for _ in range(len(ordres)-1, -1, -1):
                    if ordres[_] == 0: 
                        if df['rendement_journalier'][indice_etat+1] > 0:
                            return -df['rendement_journalier'][indice_etat+1] 
    
                        else:
                            return df['rendement_journalier'][indice_etat+1]
                        break
                    
                    
                    elif ordres[_] == 1: 
                        if df['rendement_journalier'][indice_etat+1] < 0:
                            return -df['rendement_journalier'][indice_etat+1] 
    
                        else:
                            return df['rendement_journalier'][indice_etat+1] 
                        break

    
    # Fonction de Valeur de Portefeuille (recomp2):
                    
    @staticmethod
    def recompense2(action, id_etat):
        
        global capital_actuel, ancien_capital
        
        if action in [0,1]:
            return 0
        
        elif action in [2,3]: 
            for _ in range(id_etat,-1, -1):
                
                if ordres[_] == 0: 
                    prix_entree = dt['Close'][_] 
                    prix_actuel = dt['Close'][id_etat]
                    p_l = (prix_actuel - prix_entree)*capital_actuel*fraction_investie # Profit / Perte
                    ancien_capital = capital_actuel
                    capital_actuel += p_l
                    return (capital_actuel - ancien_capital)/ancien_capital
                
                    break
                
                elif ordres[_] == 1: 
                    prix_entree = dt['Close'][_] 
                    prix_actuel = dt['Close'][id_etat]
                    p_l = (prix_entree - prix_actuel)*capital_actuel*fraction_investie
                    ancien_capital = capital_actuel
                    capital_actuel += p_l
                    return (capital_actuel - ancien_capital)/ancien_capital
                
                    break
        
        elif action == 4: # cash
            return -0.5 
        
    
    # Fonction de récompense alternative dite "sparse":
        
    def sparse(action, id_etat):
            global capital_actuel, ancien_capital
            
            if action in [0,1,2]:
                return 1
            
            elif action == 4:
                return 0
            
            elif action == 3:
                for _ in range(id_etat,-1, -1):
                    
                    if ordres[_] == 0:  # Maintien buy / buy entry
                        prix_entree = dt['Close'][_] 
                        prix_actuel = dt['Close'][id_etat]
                        p_l = (prix_actuel - prix_entree)*capital_actuel*fraction_investie
                        ancien_capital = capital_actuel
                        capital_actuel += p_l
                        return (capital_actuel - ancien_capital)/ancien_capital
                    
                        break
                    
                    elif ordres[_] == 1: 
                        prix_entree = dt['Close'][_] 
                        prix_actuel = dt['Close'][id_etat]
                        p_l = (prix_entree - prix_actuel)*capital_actuel*fraction_investie
                        ancien_capital = capital_actuel
                        capital_actuel += p_l
                        return (capital_actuel - ancien_capital)/ancien_capital
                    
                        break
            
            
                    
    # Récompense profit/perte:
                    
    @staticmethod
    def recompense3(action, id_etat):
        global capital_actuel
        if action in [0,1]:
            return 0
        
        elif action in [2,3]: 
            for _ in range(id_etat, -1, -1): 
                
                if ordres[_] == 0:  
                    prix_entree = dt['Close'][_] 
                    prix_actuel = dt['Close'][id_etat]
                    p_l = (prix_actuel - prix_entree)
                    
                    return p_l
                
                    break
                
                elif ordres[_] == 1: 
                    prix_entree = dt['Close'][_] 
                    prix_actuel = dt['Close'][id_etat]
                    p_l = (prix_entree - prix_actuel)
                    
                    return p_l
                
                    break
        
        elif action == 4: # cash
            return 0
    
    
    # Transformations utiles:
        
    #Logarithme symétrique
    @staticmethod
    def Sym_Log(x):
        if x>0:
            return np.log(x)
        elif x < 0:
            return -np.log(-x)
        else:
            return 0
    
    # Logarithme positif:
    @staticmethod
    def Pos_Log(x):
        if x>0:
            return np.log(x)
        else:
            return 0


# Paramètres:

epsilon = 1.0 # proba d'exploration
estompage = 0.9995  # facteur par lequel on estompe epsilon          
min_epsilon = 0.01 # estompe epsilon à la limite de 1%
derniere_action = 4
gamma = 0.99 # facteur d'actualisation
alpha = 0.001 # taux d'apprentissage
fraction_investie = 0.1 # investit 10% du capital

# Paramètres qui seront mis à jour par l'algorithme:
capital_actuel = 10_000
ancien_capital = 10_000


# Création de l'architecture:

def DQN():
    
    # Couches:
    init = tf.keras.initializers.glorot_normal() # Initialisation de Xavier
    inputs = tf.keras.layers.Input(shape=(10,),dtype='float64') 
    hidd_layer1 = tf.keras.layers.Dense(10, activation=tf.nn.tanh, kernel_initializer = init, dtype='float64',use_bias=True )(inputs)  
    hidd_layer2 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,  kernel_initializer = init, dtype='float64', use_bias=True )(hidd_layer1) 
    outputs = tf.keras.layers.Dense(5, activation = None, kernel_initializer = 'zeros', bias_initializer = 'zeros' , dtype='float64', use_bias=True)(hidd_layer2)

    return tf.keras.modele(inputs=inputs, outputs=outputs)


# Stocke le réseau:
modele = DQN()

# Copie du réseau pour le réseau cible
reseau_cible = DQN()

# Optimisation:
optimizer = tf.keras.optimizers.SGD(learning_rate = alpha)
#keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0) # Option alternative: algorithme ADAM

# Fonction de perte:
loss_function = keras.pertes.Huber()


# Stockage des actions, états, récompenses:

derniere_action = 4  # Etat initial: cash
histoire_etats = []
histoire_actions = []
histoire_recompenses = []
pertes = []
ordres = []
recomp_t = []
recomp_cumulative = 0
cum_recompense = []
cum_buy_q = []
histoire_capital = []


# K.clear_session() => des fois nécessaire pour réinitialiser Keras afin de déboguer.



##########################
# Boucle d'apprentissage #
##########################

while recomp_cumulative < 400 :
    
    derniere_action = 4
    recomp_cumulative = 0
    histoire_etats = []
    histoire_actions = []
    histoire_recompenses = []
    pertes = []
    ordres = []
    recomp_t = []
    cum_recompense = []
    cum_buy_q = []
    
    capital_actuel = 10_000
    ancien_capital = 10_000

    
    for indice_etat in range(0, len(df)-1):
        
        # Observe l'état actuel à partir de l'environnement:
        etat = np.matrix(Environment.etat_actuel(indice_etat))
        
        # epsilon-greedy: prends une action
        epsilon = max(epsilon, min_epsilon) # estompage epsilon
        
        if epsilon >= np.random.rand(1)[0]:
            action = np.random.choice(Environment.espace_actions(derniere_action)) # action aléatoire
            
        # Sinon prends Max Q-value (meilleure action):
        else:
            etat_tensor = tf.expand_dims(tf.convert_to_tensor(etat), axis=0)
            action_probs = modele(etat_tensor[0], training=False)
            
            # Max Q value:
            for i in Environment.espace_actions(derniere_action):
                espace_reduit = [np.array(action_probs)[0][i] for i in Environment.espace_actions(derniere_action)]
            
            for i in range(len(espace_reduit)): 
                if espace_reduit[i] == max(espace_reduit):
                    Max = espace_reduit[i]
                    
            for i in range(5):
                if np.array(action_probs)[0][i] == Max:
                    action = i
                 
        # Log des ordres:
        ordres.append(action)
        
        # Applique l'action à notre environnement:     
        # Recompense:
        action_recompense = Environment.recompense2(action, indice_etat)
        
        # Etat suivant:
        etat_suivant = np.array(Environment.etat_suivant(indice_etat))
     
        
        # Store etats, actions and recompenses:
        histoire_etats.append(etat)
        histoire_actions.append(action)
        histoire_recompenses.append(action_recompense)

        # Double DQN: Max Q
        recomp_futures = reseau_cible(tf.expand_dims(etat_suivant, axis=0))
        
        # Mise à jour Q-value: Q(s,a) = recompense + gamma * max Q(s',a'):
        valeur_q_cible = action_recompense + gamma * recomp_futures
            
        # Crée un filtre à codes 0/1 pour chaque action:
        Filter = tf.one_hot(histoire_actions, depth=5, dtype='float64')
        
        # Rétropropagation: calcul du gradient
        with tf.GradientTape() as tape:
            
            # Valeur prédite:
            pred_q_values = modele(histoire_etats)
    
            # Matrice qui donne la valeur Q pour chaque action
            q_action = tf.reduce_sum(tf.multiply(pred_q_values, Filter))
            
            # Calcul de l'erreur:
            loss = loss_function(valeur_q_cible, q_action)  
            
            
        # Arrête tout si l'erreur diverge à l'infini:
        if np.array(loss) == math.inf:
            print("Erreur à divergé:", indice_etat, "dans l'epoch':", epoch)
            break
        
        # Descente de gradient:
        grads = tape.gradient(loss, modele.trainable_variables)
        optimizer.apply_gradients(zip(grads, modele.trainable_variables))
        
        # Mise à jour des paramètres w: (poids)
        weights = modele.get_weights()
        modele.set_weights(weights)
        
        # Met à jour le réseau cible chaque 10 itérations:
        if indice_etat%10 == 0:
            reseau_cible.set_weights(modele.get_weights())
        
        # Met à jour la dernière action prise:
        derniere_action = action
        
        # estompage epsilon:
        epsilon = epsilon*estompage
        
        # Mise à jour de la récompense cumulative (condition d'arrêt)
        recomp_cumulative += action_recompense
        
        # Vide les listes:
        histoire_etats = []
        histoire_actions = []
        
        # Note les pertes:
        pertes.append(loss)
        
        # Note la récompense en fonction du temps:
        recomp_t.append(action_recompense)
        
        # Note la récompense cumulaltive dans une liste:
        cum_recompense.append(recomp_cumulative)
        
        # Log capital actuel:
        histoire_capital.append(capital_actuel)
        
        
        # Informations à défiler:
        print("--------------------------------------")
        print("- Iteration:", indice_etat)
        print("- La perte est égale à:", np.array(loss))
        print("- La probabilité d'exploration est':", epsilon)
        
    # Défile les graphes des récompenses cumulées à chaque épisode:        
    x = range(0,len(cum_recompense))
    fig, ax = plt.subplots()
    ax.plot(x, cum_recompense)
    plt.show()

'''
Fin
'''

# Diagnostics:

# Moyenne des erreurs
np.mean(pertes)

# Graphe des pertes:
x = range(0,len(pertes))
fig, ax = plt.subplots()
ax.plot(x, pertes)
plt.show()


# Graphe des récompenses en fonction du temps:
x = range(0,len(recomp_t))
fig, ax = plt.subplots()
ax.plot(x, recomp_t)
plt.show()

# Graphes des récompenses cumulatives:
x = range(0,len(cum_recompense))
fig, ax = plt.subplots()
ax.plot(x, cum_recompense)
plt.show()




###############
# Backtesting #
###############


# Transformes les codes des ordres à leurs nombs respectifs:

for i in range(len(ordres)):
    if ordres[i] == 0:
        ordres[i] = "Achat"
    
    elif ordres[i] == 1:
        ordres[i] = "Vente"
        
    elif ordres[i] == 2:
        ordres[i] = "Maintien"
    
    elif ordres[i] == 3:
        ordres[i] = "Close"
        
    elif ordres[i] == 4:
        ordres[i] = "Cloture"

# % du temps où y a eu: Maintien, Achat, Vente, cloture ou cash :
Maintien = []
Achat = []
Vente = []
Cash = []
Cloture = []

for i in range(len(ordres)):
    if ordres[i] == 'Maintien':
        Maintien.append(ordres[i])
        
    if ordres[i] == 'Achat':
        Achat.append(ordres[i])
    
    if ordres[i] == 'Vente':
        Vente.append(ordres[i])
    
    if ordres[i] == 'Cash':
        Cash.append(ordres[i])
        
    if ordres[i] == 'Cloture':
        Close.append(ordres[i])

    
print("- % du temps passé en Maintien:" , len(Maintien)/len(ordres)*100)
print("- % du temps passé en Achat:" , len(Achat)/len(ordres)*100)
print("- % du temps passé en Vente:" , len(Vente)/len(ordres)*100)
print("- % du temps passé en Cash:" , len(Cash)/len(ordres)*100)
print("- % du temps passé en Cloture:" , len(Close)/len(ordres)*100)
print("")
print("- Nombre des achats:", ordres.count('Achat'))
print("- Nombre des ventes:",ordres.count('Vente'))
print("- Total des ordres:", ordres.count('Achat') + ordres.count('Vente'))


# Donnees OHLC:
ohlc = dt['Close'][0: len(dt)]
ohlc = ohlc.reset_index(drop= True)


# Ordres:
lsch = pd.DataFrame(ordres)


# Check si même taille:
len(ohlc) == len(lsch)


# Combine les deux vecteurs:
bt_dt = pd.concat([ohlc, lsch], axis=1)


# Courbe du capital:
Capital_debut = 10_000
pourcent_risque = 0.1
taille_ordre = Capital_debut*pourcent_risque
courbe_capital = [Capital_debut]


# Closes the very last positon if left open:
if bt_dt[0][len(bt_dt)-1] in ["Achat","Vente","Maintien"]:
    bt_dt[0][len(bt_dt)-1] = 'Cloture'
    

# Backtester:
for i in range(len(bt_dt)):
    if bt_dt[0][i] == 'Achat':
        entree = bt_dt['Close'][i]  # Prix d'entrée
    
        
        for o in range(i, len(bt_dt)):

            if bt_dt[0][o] == 'Close':
                sortie = bt_dt['Close'][o] # Stocke prix de sortie
                p_l = (sortie - entree)*taille_ordre # profit/perte
                Capital_debut += p_l
                courbe_capital.append(Capital_debut)
                break

            
    elif bt_dt[0][i] == 'Short':

        entree = bt_dt['Close'][i]  # stocke prix entree
    
        for o in range(i, len(bt_dt)):  # Check où est la cloture

            if bt_dt[0][o] == 'Close':
                sortie = bt_dt['Close'][o] 
                p_l = (entree - sortie)*taille_ordre  # Profit/Perte
                Capital_debut += p_l
                courbe_capital.append(Capital_debut)
                break
        

# Visualise la courbe de capital final:
x = range(0,len(courbe_capital))
fig, ax = plt.subplots()
ax.plot(x, courbe_capital)
ax.set_xlabel('Transactions')
ax.set_ylabel('Capital')
ax.set_title(r"Evolution du capital de l'agent après chaque transaction")
plt.show()



###############
# Performance #
###############

# Performance Stratégie:

print("- Retour sur l'investissement de la stratégie(%):",((courbe_capital[len(courbe_capital)-1] / courbe_capital[0])-1)*100)


# Performance Buy & Hold:

BH_profit = (bt_dt['Close'][len(bt_dt['Close'])-1] - bt_dt['Close'][0])*10000*pourcent_risque
print("- Retour sur l'investissement Buy & Hold (%):",(((BH_profit+10000)/10000)-1)*100) 