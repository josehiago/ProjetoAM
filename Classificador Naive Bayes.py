# -*- coding: utf-8 -*-
from __future__ import division
import collections
import math

class Model:
        def __init__(self, arffFile):
                self.trainingFile = arffFile
                self.features = {} # todos os nomes das funções e seus possíveis valores (incluindo o rótulo da classe)
                self.featureNameList = [] # é para manter a ordem dos recursos como no arff
                self.featureCounts = collections.defaultdict(lambda: 1)# contém as tuplas do formulário (rótulo, feature_name, feature_value)
                self.featureVectors = [] # contém todos os valores eo rótulo como a última entrada
                self.labelCounts = collections.defaultdict(lambda: 0) # estes serão suavizados mais tarde (armazena as contagens dos próprios rótulos da classe)

        def TrainClassifier(self):#conta o número de co-ocorrências de cada valor de recurso com cada rótulo de classe e os armazena na forma de 3-tuplas. Essas contagens são suavizadas automaticamente usando o alinhamento de adicionar-um como o valor padrão de contagem para este dicionário é '1'. As contagens dos rótulos também são ajustadas ao incrementar essas contagens pelo número total de observações.
                for fv in self.featureVectors:
                        self.labelCounts[fv[len(fv)-1]] += 1 # udpate contagem do rótulo
                        for counter in range(0, len(fv)-1):
                                self.featureCounts[(fv[len(fv)-1], self.featureNameList[counter], fv[counter])] += 1

                for label in self.labelCounts: ## aumenta a contagem de rótulos (alisamento). Lembre-se de que o último recurso é realmente o rótulo
                        for feature in self.featureNameList[:len(self.featureNameList)-1]:
                                self.labelCounts[label] += len(self.features[feature])

        def Classify(self, featureVector): ## featureVector é uma lista simples como as que usamos para treinar
                probabilityPerLabel = {} # armazenar a probabilidade final para cada etiqueta de classe
                for label in self.labelCounts: #como argumento, um vetor de recurso único (como uma lista) e calcula o produto de probabilidades condicionais individuais (MLE suavizado) para cada rótulo. As probabilidades calculadas finais para cada rótulo são armazenadas no dicionário ' probabilidadeListaLabel '. Na última linha, devolvemos a entrada de probabilidadPerLabel que tem maior probabilidade. Note-se que a multiplicação é realmente feita como adição no domínio do log, pois os números envolvidos são extremamente pequenos. Além disso, um dos fatores usados ​​nesta multiplicação, é a probabilidade anterior de ter esse rótulo de classe.
                        logProb = 0
                        for featureValue in featureVector:
                                logProb += math.log(self.featureCounts[(label, self.featureNameList[featureVector.index(featureValue)], featureValue)]/self.labelCounts[label])
                        probabilityPerLabel[label] = (self.labelCounts[label]/sum(self.labelCounts.values())) * math.exp(logProb)
                print probabilityPerLabel
                return max(probabilityPerLabel, key = lambda classLabel: probabilityPerLabel[classLabel])
                                
        def GetValues(self):#lê os nomes das funções (incluindo os rótulos das classes), seus possíveis valores e os próprios vetores de características; e preencha as estruturas de dados
                file = open(self.trainingFile, 'r')
                for line in file:
                        if line[0] != '@': ## início de dados reais
                                self.featureVectors.append(line.strip().lower().split(','))
                        else: #feature definitions
                                if line.strip().lower().find('@data') == -1 and (not line.lower().startswith('@relation')):
                                        self.featureNameList.append(line.strip().split()[1])
                                        self.features[self.featureNameList[len(self.featureNameList) - 1]] = line[line.find('{')+1: line.find('}')].strip().split(',')
                                       #self.features[self.featureNameList[len(self.featureNameList) - 1]] = [featureName.strip() for featureName in line[line.find('{')+1: line.find('}')].strip().split(',')]  
                file.close()

        def TestClassifier(self, arffFile):
                file = open(arffFile, 'r')
                count_acertos = 0
                count_class = 0
                for line in file:
                        if line[0] != '@':
							
                                vector = line.strip().lower().split(',')	
                                count_class = count_class + 1
                                if str(self.Classify(vector)) == str(vector[len(vector) - 1]):
									count_acertos = count_acertos + 1		
								
                                print "classifier: " + self.Classify(vector) + " given " + vector[len(vector) - 1]
				print "Acurácia: " + str("%.2f" % round((count_acertos/count_class)*100,2))+ " %"

if __name__ == "__main__":
        model = Model(r"C:\Users\hiago\Documents\train.arff.norm.arff")
        model.GetValues()
        model.TrainClassifier()
        model.TestClassifier(r"C:\Users\hiago\Documents\test.arff.norm.arff")

#https://sites.google.com/site/carlossillajr/
