#variables derived from detection phase
mu_t = None #truthful centroid 
mu_h = None #false/hallucinated centroid
h_l = None #normalized hidden vector
tsv = None #tsv vector
confidence_score = None #hallucination confidence score

#hyperparameters 
beta = None
alpha = None

#first outlined mitigation method
def prototype_interpolation():
    adjusted_representation = (1 - beta)*h_l + beta*(mu_t)
    return adjusted_representation

#second outlined mitigation method:
def adaptive_mitigation():
    adjusted_representation = h_l + confidence_score*alpha*tsv
    return adjusted_representation

#third outlined mitigation method
def prototype_aware_projection():
    adjusted_representation = h_l + alpha*(mu_t - mu_h)
    return adjusted_representation
