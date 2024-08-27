''' 
EGA-FMC: Enhanced Genetic Algorithm based Fuzzy K-Modes Clustering for Categorical Data
https://dl.acm.org/doi/10.1504/IJBIC.2018.092801
https://github.com/medhini/Genetic-Algorithm-Fuzzy-K-Modes
'''
import numpy as np
import random
from collections import defaultdict

def dissimilarityMeasure(X, Y): 
    """ Simple matching disimilarity measure """
    return np.sum(X != Y, axis=0)
    
def separation(centroids, membership_mat):
    sep = 0.0
    k = len(centroids)
    for x in range(k):
        for y in range(x, k, 1):
            sep += np.power(membership_mat[x][y], alpha) * dissimilarityMeasure(centroids[x], centroids[y])
    return sep

"""Compactness or CostFunction"""
def costFunction(membership_mat, n_clusters, n_points, alpha, centroids, X_Features):
    cost_function = 0.0
    
    for k in range(n_clusters):
        temp = 0.0
        denom = 0.0
        for i in range(n_points):
            temp += np.power(membership_mat[i][k], alpha) * dissimilarityMeasure(X_Features[i], centroids[k])
            denom += np.power(membership_mat[i][k], alpha)
        temp = temp / denom
        cost_function += temp

    return cost_function

def updateMatrix(centroids, X_Features, n_points, n_clusters, n_attributes, alpha):
    membership_mat = np.zeros((n_points, n_clusters))  # Ensure it's initialized here
    exp = 1 / (float(alpha - 1))
    
    for x in range(n_clusters):
        centroid = centroids[x]
        for y in range(n_points):
            hammingDist = dissimilarityMeasure(centroid, X_Features[y])
            numerator = np.power(hammingDist, exp)
            denom = 0.0
            flag = 0
            
            for z in range(n_clusters):
                if (centroids[z] == X_Features[y]).all() and (centroids[z] == centroid).all():
                    membership_mat[y][x] = 1
                    flag = 1
                    break
                elif (centroids[z] == X_Features[y]).all():
                    membership_mat[y][x] = 0
                    flag = 1
                    break

                denom += np.power(dissimilarityMeasure(centroids[z], X_Features[y]), exp)
        
            if flag == 0:
                membership_mat[y][x] = 1 / (float(numerator) / float(denom))         
            
    for row in range(len(membership_mat)):
        membership_mat[row] = membership_mat[row] / sum(membership_mat[row])

    cost_function = costFunction(membership_mat, n_clusters, n_points, alpha, centroids, X_Features)
    return membership_mat, cost_function

def calculateCentroids(membership_mat, X_Features, alpha):
    n_points, n_attributes = X_Features.shape
    n_clusters = membership_mat.shape[1]

    WTemp = np.power(membership_mat, alpha)
    centroids = np.zeros((n_clusters, n_attributes))

    for z in range(n_clusters):
        for x in range(n_attributes):
            freq = defaultdict(int)
            for y in range(n_points):
                freq[X_Features[y][x]] += WTemp[y][z]

            centroids[z][x] = max(freq, key=freq.get)
    
    centroids = centroids.astype(int)

    return centroids

def fuzzyKModes(membership_mat, X_Features, alpha, max_epochs):
    n_points, n_clusters = membership_mat.shape
    n_attributes = X_Features.shape[1]

    centroids = np.zeros((n_clusters, n_attributes))
    epochs = 0
    oldCostFunction = 0.0
    costFunctionValue = 0.0

    while epochs < max_epochs:
        centroids = calculateCentroids(membership_mat, X_Features, alpha)
        membership_mat, costFunctionValue = updateMatrix(centroids, X_Features, n_points, n_clusters, n_attributes, alpha)

        if (oldCostFunction - costFunctionValue) * (oldCostFunction - costFunctionValue) < 0.3:
            break
        oldCostFunction = costFunctionValue
        epochs += 1
    
    return membership_mat, costFunctionValue

def Selection(chromosomes, n, k):
    """Rank Based Fitness Assignment"""
    
    # Sort chromosomes for rank based evaluation
    chromosomes = chromosomes[chromosomes[:, n * k].argsort()]
    newChromosomes = np.zeros((n, n * k + 1))

    beta = 0.1
    fitness = np.zeros(n)
    cumProbability = np.zeros(n)

    for i in range(n - 1, 0, -1):
        fitness[i] = beta * (pow((1 - beta), i))

    """Roulette Wheel Selection"""

    # Cumulative Probability
    for i in range(n):
        if i > 1:
            cumProbability[i] = cumProbability[i - 1] 
        cumProbability[i] += fitness[i]
    
    # Random number to pick chromosome
    for i in range(n):
        pick = random.uniform(0, 1)

        if pick < cumProbability[0]:
            newChromosomes[i] = chromosomes[0]
        else:    
            for j in range(n - 1):
                if cumProbability[j] < pick < cumProbability[j + 1]:
                    newChromosomes[i] = chromosomes[j + 1]
        
        newChromosomes[i][n * k] = 0.0
    
    return newChromosomes

def CrossOver(chromosomes, n, k, X_Features, alpha):
    newChromosomes = np.zeros((n, n * k + 1))
    
    for i in range(n):
        membership_mat = np.reshape(chromosomes[i][0:n * k], (-1, k))
        new_membership_mat, cost_function = fuzzyKModes(membership_mat, X_Features, alpha, 1)    # Quick termination, 1 step fuzzy kmodes
        newChromosomes[i][0 : n * k] = new_membership_mat.ravel()
        newChromosomes[i][n * k] = cost_function

    return newChromosomes

def Mutation(chromosomes, n, k):
    P = 0.001
    for i in range(n):
        chromosome = chromosomes[i][0 : n * k]
        chromosome = np.reshape(chromosome, (-1, k))

        for j in range(n):
            pick = random.uniform(0, 1)
            if pick <= P:
                gene = np.random.rand(k)
                gene = gene / sum(gene)
                chromosome[j] = gene

        chromosomes[i][0 : n * k] = chromosome.ravel()

    return chromosomes

def ega_fmc(x, k):
    X_Features = x[:, :].astype(int)
    n = len(X_Features)
    alpha = 1.2
    max_epochs = 100
    g_max = 15

    populationSize = n
    chromosomes = np.zeros((n, n * k + 1))

    # print("GA-FKM start")

    """Initialize Population"""
    for i in range(populationSize):
        membership_mat = np.random.rand(n, k)

        for row in range(len(membership_mat)):
            membership_mat[row] = membership_mat[row] / sum(membership_mat[row])

        chromosomes[i][0 : n * k] = membership_mat.ravel()

        centroids = calculateCentroids(membership_mat, X_Features, alpha)
        chromosomes[i][n * k] = costFunction(membership_mat, k, n, alpha, centroids, X_Features)
            
    """Genetic Algorithm K Modes"""
    for x in range(g_max):
        """Best parent of this generation"""
        min_value = 0
        best_parent = chromosomes[0]
        for i in range(populationSize):
            if min_value == 0:
                min_value = chromosomes[i][n * k]

            elif chromosomes[i][n * k] < min_value:
                min_value = chromosomes[i][n * k]
                best_parent = chromosomes[i]
        
        population_after_selection = Selection(chromosomes, n, k)
        population_after_crossover = CrossOver(population_after_selection, n, k, X_Features, alpha)
        chromosomes = Mutation(population_after_crossover, n, k)

        """Elitism at each generation"""
        max_value = 0
        worst_child_pos = 0
        for i in range(populationSize):
            membership_mat = np.reshape(chromosomes[i][0:n * k], (-1, k))
            centroids = calculateCentroids(membership_mat, X_Features, alpha)
            chromosomes[i][n * k] = costFunction(membership_mat, k, n, alpha, centroids, X_Features)
            if max_value == 0:
                max_value = chromosomes[i][n * k]

            elif chromosomes[i][n * k] > max_value:
                max_value = chromosomes[i][n * k]
                worst_child_pos = i

        chromosomes[worst_child_pos] = best_parent

    """Best of the child chromosomes"""
    min_value = 0
    offspring = chromosomes[0]

    for i in range(populationSize):
        if min_value == 0:
            min_value = chromosomes[i][n * k]

        elif chromosomes[i][n * k] < min_value:
            min_value = chromosomes[i][n * k]
            offspring = chromosomes[i]
    final_membership_mat = np.reshape(offspring[0:n*k], (n, k))
    cluster_labels = np.argmax(final_membership_mat, axis=1)
    return cluster_labels