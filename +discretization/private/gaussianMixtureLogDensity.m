function value = gaussianMixtureLogDensity(x,p,means,sds)
active = p > 0;
p = p(active); means = means(active); sds = sds(active);
terms = log(p(:)')-log(sds(:)')-0.5*log(2*pi) ...
    -0.5*((x(:)-means(:)')./sds(:)').^2;
largest = max(terms,[],2);
value = largest+log(sum(exp(terms-largest),2));
end
