%% Kuramoto-Sivashinsky equation and chaos
nn = 511;
steps = 250;

dom = [-1 1]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) - 0.5 * diff(u,2) - 0.005 * diff(u,4);
S.nonlin = @(u) - 5 * 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
% S.init = cos(x/16).*(1+sin(x/16));
S.init = -sin(pi*x);
u = spin(S,nn,1e-5, 'plot', 'off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1,1,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('ks.mat','t','x','usol')