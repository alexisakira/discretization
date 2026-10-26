function generateReferenceData
% Regenerate 0.2.0 process fixtures. Historical 0.1.0 fixtures stay immutable.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
f = load(fullfile(root,'reference_data','matlab_r2024b_baseline.mat'));
[f.arTransition,f.arGrid] = discretization.discreteAR(f.arMu,f.arRho, ...
    f.arSigma,f.arStateCount,f.arMethod,f.arMomentCount);
[f.ar4Transition,f.ar4Grid] = discretization.discreteAR(0,.9,.1,9,'even',4);
[f.varTransition,f.varGrid] = discretization.discreteVAR(zeros(2,1), ...
    [.9809 .0028;.0410 .9648],diag([.0087^2,.0262^2]),9,2,'even');
mixture = {[.1628,.8372],[-.0039,.0008],[.1293,.0300]};
[f.gmar1Transition,f.gmar1Grid,f.gmar1Diagnostics] = ...
    discretization.discreteGaussianMixtureAR(.0555,.5854,mixture{:},9,2,'even');
[f.gmar2Transition,f.gmar2Grid,f.gmar2Diagnostics] = ...
    discretization.discreteGaussianMixtureAR(.0555,[.8959,-.3990],mixture{:},9,2,'even');
[f.svTransition,f.svGrid] = discretization.discreteStochasticVolatilityAR(.95,.9,.007,.06,9,5);
[f.cirTransition,f.cirGrid] = discretization.discreteCIR(-log(.58),.028,.1,.25,9);
[f.ouTransition,f.ouGrid] = discretization.discreteOU(.7,.3,.25,21);
f.matlabVersion = version;
f.matlabRelease = version('-release');
f.packageVersion = '0.2.0';
methods = {'even','gauss-legendre','clenshaw-curtis','gauss-hermite','GMQ'};
for index = 1:numel(methods)
    [f.(['gmOptionalP',num2str(index)]),f.(['gmOptionalX',num2str(index)]), ...
        f.(['gmOptionalD',num2str(index)])] = discretization.discreteGaussianMixtureAR( ...
        0,.4,[.4,.6],[-.1,.1],[.1,.2],21,4,methods{index},4);
end
[f.gmSkewP,f.gmSkewX,f.gmSkewD] = discretization.discreteGaussianMixtureAR( ...
    0,.95,[.05,.95],[-.8,.8*.05/.95],[.2,.03],31,4,'even');
for method = {'quantile','quadrature'}
    [f.(['var',method{1},'P']),f.(['var',method{1},'X'])] = ...
        discretization.discreteVAR(0,.4,.01,21,4,method{1});
end
save(fullfile(root,'reference_data','matlab_r2026a_0_2.mat'),'-struct','f','-v7');
end
