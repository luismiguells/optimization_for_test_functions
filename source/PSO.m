clear; clc; close all;

%% Test functions for optimization
% Ackley function
% -20*exp(-0.2*sqrt(0.5*(x.^2 + y.^2))) - exp(0.5*(cos(2*pi*x)+cos(2*pi*y))) + exp(1) + 20 

% Beale function
% (1.5-x+(x*y)).^2 + (2.25*-x+(x*y.^2)).^2 + (2.625-x+(x*y.^3)).^2

% Booth function
% (x+(2*y)-7).^2 + ((2*x)+y-5).^2   

% Himmelblau's function
% (x.^2+y-11).^2 + (x+y.^2-7).^2

% McCormick function
% sin(x+y) + (x+y).^2 - 1.5*x + 2.5*y + 1

% More in: https://en.wikipedia.org/wiki/Test_functions_for_optimization

%% Initialization
a = -5; b = 5; N = 40;
X = b - (b-a).*rand(N,2); % Random position of particles
iter = 1;
max_iter = 200;  
neighbors = 5;
Y = 0; % Target value
epsilon = 1e-6;

%% PSO parameters
C1 = 0.5;          
C2 = 0.5;          
V = 0.05;
W = 0.5;


%% Begin algorithm
while iter < max_iter
    loc = objective_function(X);
    [q, indx_0] = sort(loc,1,'ascend');
    g_best = X(indx_0(1),:);
    for k=1:N
        dist = sqrt(sum((X(k,:)- X).^2, 2));
        [q, indx] = sort(dist, 1, 'ascend');
        X_N = X(indx(1:neighbors,:),:);  % Besr neighbors
        L_N = loc(indx(1:neighbors,:),:);  % Best localization of neighbors
        [q, indx1] = sort(L_N, 1, 'ascend');
        p_best = X_N(indx1(1),:);
        V = W*V + C1.*rand().*(p_best-X(k,:)) + C2.*rand().*(g_best-X(k,:));
        X(k,:)= X(k,:) + V;   % Update the position
    end
    
    % Calculate the error
    error = abs(objective_function(g_best(1,:)) - Y);
    
    if(error < epsilon)
        break
    end
    
    iter = iter + 1;
end
 
fprintf('The minimum value is in (X, Y): (%f, %f) in %d iterations\n', g_best(1,1), g_best(1,2), iter);

% Plot the function
[X,Y] = meshgrid(a:b);                                
Z = (X+(2*Y)-7).^2 + ((2*X)+Y-5).^2; % Change for the desire function

mesh(X, Y, Z);
colormap(gray)
hold on;
scatter(g_best(1,1), g_best(1,2), 100, 'filled', 'r');

%% Objective function
function Y = objective_function(X)        
   x = X(:, 1); 
   y = X(:, 2);
   Y = (x+(2*y)-7).^2 + ((2*x)+y-5).^2; % Change for the desire function
end
