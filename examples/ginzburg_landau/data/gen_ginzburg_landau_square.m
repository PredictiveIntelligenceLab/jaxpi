
k = 10;
eps = k / (50 * 50);

% k = 15;
% eps = 1e-3;

dom = [-1 1 -1 1];

steps = 100;
t = linspace(0, 1, steps+1);

S = spinop2(dom,t);
S.lin = @(u) eps * lap(u);
S.nonlin = @(u) k * u - k * (1+1.5i)*u.*(abs(u).^2);

x = chebfun2(@(x,y) x,dom); y = chebfun2(@(x,y) y,dom);
u1 = (1i*x * 10 + y * 10).*exp(-.01*(x.^2 * 2500 +y.^2 * 2500)); S.init = u1;

npts = 200; dt = 1/npts/5 ; tic
u = spin2(S,npts,dt,'plot','off');

% u = spin2(S,npts,dt);

figure(1)
plot(real(u{101})), view(0,90), axis equal, axis off

figure(2)
plot(imag(u{101})), view(0,90), axis equal, axis off

N = 200;
[X,Y] = meshgrid(linspace(-1,1, N), linspace(-1,1, N));

usol = zeros(steps+1, N, N);
vsol = zeros(steps+1, N, N);

for i = 1:steps+1
    usol(i,:,:) = real(u{1, i}(X,Y));
    vsol(i,:,:) = imag(u{1, i}(X,Y));
end

x = linspace(-1,1, N);
y = linspace(-1,1, N);
% 
save('ginzburg_landau_square.mat', 'eps', 'k', 'usol', 'vsol', 't', 'x', 'y')
