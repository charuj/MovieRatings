
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special
import scipy.misc
import scipy.stats

# Loading dataset
import csv
with open('u.data', 'rb') as csvfile:
    movie_rate = csv.reader(csvfile, delimiter='\t', quotechar='|')

    all_data= []
    for row in movie_rate:
       all_data.append(row)
    all_data=np.array(all_data)

ratings= all_data[:, 2]
ratings_length= ratings.shape[0]
ratings_train=ratings[0:ratings_length * 0.8]
ratings_test= ratings[ratings_length*0.8:]


ratings_train= ratings_train.astype(float)
ratings_test= ratings_test.astype(float)

N= ratings_train.shape[0]
N= N +0.0

N_test=ratings_test.shape[0]
N_test= N_test + 0.0


# plot marginal distribution of ratings
# rect_histx = [1, 0, 4, ratings.shape[0]]
# axHistx = plt.axes(rect_histx)
# bins = np.arange(1 ,5, 1)
# axHistx.hist(ratings, bins=bins)


zero=[0,0,0,0,0]
for value in ratings_train:
    value= int(value)
    rate= value - 1
    zero[rate]= zero[rate]+1
print ratings_train
# n_groups = 5
# index = np.arange(n_groups)
# bar_width = 0.35
#
# rects1 = plt.bar(index,  bar_width)
# plt.show()


#plt.bar([0,1,2,3,4], zero, width=1)
#plt.show()



# priors on test test

zero_test= [0,0,0,0,0]
for val in ratings_test:
    val= int(val)
    rate_test= val - 1
    zero_test[rate_test]= zero_test[rate_test]+1
print ratings_test

prior1_test= zero_test[0]/N_test
prior2_test= zero_test[1]/N_test
prior3_test= zero_test[2]/N_test
prior4_test= zero_test[3]/N_test
prior5_test= zero_test[4]/N_test



# Gaussian: max likelihood

mean= np.mean(ratings_train)
variance= np.std(ratings_train)
variance= variance **(1/2)

print "Mean = " + str(mean)
print "Variance " + str(variance)


# pdf of univariate normal >>> this isn't working

exponent1= math.exp((-1*((1-mean)**2)/(2*(variance**2))))
#pdf1= (1/(variance *((2)**1/(2)*math.pi ) )*exponent1)
denom1= (variance *((2*math.pi))**(1/2))
pdf1= exponent1/ (variance *(((2)*math.pi)**(1/2)))

exponent2= math.exp((-1/(2*variance**2)*((2-mean)**2)))
denom2= (variance *(((2)*math.pi)**(1/2)))
pdf2= exponent2/ (variance *(((2)*math.pi)**(1/2)))

exponent3= math.exp((-1*((3-mean)**2)/(2*(variance**2))))
denom3= (variance *((2*math.pi))**(1/2))
pdf3= exponent3/ (variance *(((2)*math.pi)**(1/2)))

exponent4= math.exp((-1*((4-mean)**2)/(2*(variance**2))))
denom4= (variance *((2*math.pi))**(1/2))
pdf4= exponent4/ (variance *(((2)*math.pi)**(1/2)))

exponent5= math.exp((-1*((5-mean)**2)/(2*(variance**2))))
denom5= (variance *((2*math.pi))**(1/2))
pdf5= exponent5/ (variance *(((2)*math.pi)**(1/2)))

# log probs of gaussian

logpdf1= np.log(prior1_test/pdf1)
logpdf2= np.log(prior2_test/pdf2)
logpdf3= np.log(prior3_test/pdf3)
logpdf4= np.log(prior4_test/pdf4)
logpdf5= np.log(prior5_test/pdf5)


print "logpdf1= " + str(logpdf1)
print "logpdf2= " + str(logpdf2)
print "logpdf3= " + str(logpdf3)
print "logpdf4= " + str(logpdf4)
print "logpdf5= " + str(logpdf5)
# beta binomial

# Step 1: finding moments, equations from Wikipedia

# find mean

m1= np.mean(ratings_train)
m2= np.mean(ratings_train **2)

# finding alpha and beta

n_betab=4

alpha= (n_betab*m1 - m2)/ (n_betab*(m2/m1 - m1 - 1)+m1)  # Changed N to 4, see piazza
print "Alpha= " + str(alpha)

beta= ((n_betab-m1)*(n_betab-m2/m1))/ (n_betab*(m2/m1 - m1 - 1) +m1) # changed N to 4, see piazza

print "Beta= " + str(beta)

# Step 2: finding the denominator beta function

betafunction= scipy.special.beta(alpha, beta)

# Step 3: numerators



prior1= zero[0]/N  # for some reason this was zero
prior2= zero[1]/N
prior3= zero[2]/N
prior4= zero[3]/N
prior5= zero[4]/N


total_priors= prior1 + prior2 + prior3 + prior4 + prior5  # this equals 1.0, which makes sense

p1= ((prior1)**(alpha -1))*((1-prior1)**(beta-1))/betafunction
p2= ((prior2)**(alpha -1))*((1-prior2)**(beta-1))/betafunction
p3=((prior3)**(alpha -1))*((1-prior3)**(beta-1))/betafunction
p4= ((prior4)**(alpha -1))*((1-prior4)**(beta-1))/betafunction
p5= ((prior5)**(alpha -1))*((1-prior5)**(beta-1))/betafunction

total_p= p1 +p2 +p3+p4 +p5 # this does not equal 1.0 ... why

# Step 4: plugging into binomial distribution

# rating1= scipy.misc.comb(N, zero[0])

rating1= scipy.misc.comb(4, 0)*(p1**0)*((1- p1)**(4))
rating2= scipy.misc.comb(4, 1)*(p2**1)*((1- p2)**(3))
rating3= scipy.misc.comb(4, 2)*(p3**2)*((1- p3)**(2))
rating4= scipy.misc.comb(4, 3)*(p4**3)*((1- p4)**(1))
rating5= scipy.misc.comb(4, 4)*(p5**4)*((1- p5)**(0))

total_rating= rating1 + rating2 + rating3 + rating4 + rating5 # this does not equal 1.0
print "Total rating prob= " + str(total_rating)

plt.plot([rating1, rating2, rating3, rating4, rating5])
plt.show()

log_pdf1_bb= np.log(prior1_test/rating1)
log_pdf2_bb= np.log(prior2_test/rating2)
log_pdf3_bb= np.log(prior3_test/rating3)
log_pdf4_bb= np.log(prior4_test/rating4)
log_pdf5_bb= np.log(prior5_test/rating5)

print "log pdf1_bb= " +str(log_pdf1_bb)
print "log pdf2_bb= " +str(log_pdf2_bb)
print "log pdf3_bb= " +str(log_pdf3_bb)
print "log pdf4_bb= " +str(log_pdf4_bb)
print "log pdf5_bb= " +str(log_pdf5_bb)
