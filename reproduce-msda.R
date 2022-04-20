library(msda)
library(Matrix)
library(MASS)

# model 1 #
Repeat = 500
error_rate = matrix(data=0, nrow=1, ncol=Repeat)
accuracy = matrix(data=0, nrow=1, ncol=Repeat)

for (t in 1:Repeat) {
  K = 4                                                  # number of classes #
  n_by_class = 75
  n = n_by_class * K                                     # sample size #   
  P = 800                                                # dimension #
  rho = 0.5                                              # AR(rho) parameter #
  n_test_by_class = 250
  n_test = n_test_by_class * K                           # test set size
  
  beta = matrix(data=0, nrow=P, ncol=K)                  # generate beta #
  for (k in 1:K) {
    beta[2*k-1, k] = 1.6
    beta[2*k, k] = 1.6
  }
  
  Sigma = matrix(data=0, nrow=P, ncol=P)                 # variance matrix #
  for (j in 1:P) {
    for (k in 1:P) {
      Sigma[j, k] = rho^abs(j-k)
    }
  }
  
  mu = matrix(data=0, nrow=P, ncol=K)                    # mean matrix $
  mu = Sigma %*% beta                                    
  
  set.seed(3*t-2)
  
  X_train = matrix(data=0, nrow=P, ncol=n)                  # generate train set $
  Y_train = matrix(data=0, nrow=1, ncol=n)
  for (k in 1:K) {
    temp = t(as.matrix(mvrnorm(n_by_class, mu[, k], Sigma)))
    for (p in 1:P){
      for (i in 1:n_by_class) {
        X_train[p, (k-1)*n_by_class+i] = temp[p, i]
      }
    }
    for (i in 1:n_by_class) {
      Y_train[(k-1)*n_by_class+i] = k
    }
  }
  Y_train = as.factor(Y_train)
  
  set.seed(3*t-1)
  
  X_valid = matrix(data=0, nrow=P, ncol=n)                  # generate validation set $
  Y_valid = matrix(data=0, nrow=1, ncol=n)
  for (k in 1:K) {
    temp = t(as.matrix(mvrnorm(n_by_class, mu[, k], Sigma)))
    for (p in 1:P){
      for (i in 1:n_by_class) {
        X_valid[p, (k-1)*n_by_class+i] = temp[p, i]
      }
    }
    for (i in 1:n_by_class) {
      Y_valid[(k-1)*n_by_class+i] = k
    }
  }
  Y_valid = as.factor(Y_valid)
  
  set.seed(3*t)
  
  X_test = matrix(data=0, nrow=P, ncol=n_test)                     # generate test set $
  Y_test = matrix(data=0, nrow=1, ncol=n_test)
  for (k in 1:K) {
    temp = t(as.matrix(mvrnorm(n_test_by_class, mu[, k], Sigma)))
    for (p in 1:P){
      for (i in 1:n_test_by_class) {
        X_test[p, (k-1)*n_test_by_class+i] = temp[p, i]
      }
    }
    for (i in 1:n_test_by_class) {
      Y_test[(k-1)*n_test_by_class+i] = k
    }
  }
  Y_test = as.factor(Y_test)
  
  # use package "msda" #
  params = cv.msda(t(X_valid), t(Y_valid), nfolds = 5, lambda = NULL, lambda.opt = "min")
  lambda = params$lambda.min
  model = msda(t(X_train), t(Y_train), lambda=lambda)
  Y_predict = as.factor(predict(model, t(X_test)))
  
  count = 0
  for (i in 1:n_test) {
    if (Y_predict[i] != Y_test[i]) {count = count + 1}
  }
  error_rate[t] = count / n_test
  accuracy[t] = (n_test - count) / n_test
}

err = mean(error_rate)
stdvar = var(error_rate)




