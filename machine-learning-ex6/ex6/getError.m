function [err] = getError(X, y, Xval, Yval, C, sigma)

m = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
pred = svmPredict( m, Xval);
err = mean(double(pred ~= Yval));

end
