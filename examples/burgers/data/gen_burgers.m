%% Burgers equation and chaos
nn = 511;
steps = 200;
nu = 0.01 / pi;

dom = [-1 1]; x = chebfun('x',dom); t = linspace(0,1,steps+1);
S = spinop(dom, t);
S.lin = @(u) + nu*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = -sin(pi*x);
u = spin(S, nn, 1e-5, 'plot', 'off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1,1,nn+1);
usol = [usol;usol(1,:)];
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('burgers.mat','t','x', 'nu', 'usol');
