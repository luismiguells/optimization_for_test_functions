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
generations = 1000;
population_size = 40;
upper_limit = -10;
lower_limit = 10;
gen_ite = 1;
Y = 0; % Target value
epsilon = 1e-6;


%% Initialize the population
population = generate_population(population_size, upper_limit, lower_limit);

%% Begin algorithm
while gen_ite < generations
    
    population = make_next_generation(population, population_size);
    best_individual = sort_population_by_function(population);
    
    % Calculate the error
    error = abs(objective_function(best_individual(1,:)) - Y);
    
    if(error < epsilon)
        break
    end

    gen_ite = gen_ite + 1; 
end

fprintf('The minimum value is in (X, Y): (%f, %f) in %d generations\n', best_individual(1,1), best_individual(1,2), gen_ite);

% Plot the function
[X,Y] = meshgrid(upper_limit:lower_limit);                                
Z = (X+(2*Y)-7).^2 + ((2*X)+Y-5).^2; % Change for the desire function

mesh(X, Y, Z);
colormap(gray)
hold on;
scatter(best_individual(1,1), best_individual(1,2), 100, 'filled', 'r');

%% Generates a random population
function population = generate_population(population_size, upper_limit, lower_limit)
    population = lower_limit - (lower_limit-upper_limit).*rand(population_size, 2);
end

%% Evaluate the objective function with each individual
function Y = objective_function(individual)
    x = individual(:,1);
    y = individual(:,2);
    
    Y = (x+(2*y)-7).^2 + ((2*x)+y-5).^2; % Change for the desire function
end

%% Use the roulette method for selection
function indi = choice_by_roulette(sorted_population, objective_sum)
    number = 1*rand(1,1);
    
    sub_sum = 0;
    N = length(sorted_population);
    
    % Calculate the probability of each individual and add it until the 
    % the cumulative probability is greater than the number chosen at random
    for i=1:N
        probability = 1.0 - (objective_function(sorted_population(i,:)) /objective_sum);
        sub_sum = sub_sum + probability;
        if number <= sub_sum
            indi = sorted_population(i,:);
            break
        end
    end
end

%% Sort the population by the result of the objective function
function pop_sorted = sort_population_by_function(population)
    result_objective = objective_function(population);
    population = [population result_objective];
    pop_sorted = sortrows(population, 3, 'ascend');
end

%% Crossover the best two individuals
function new_individual_c = crossover(individual_a, individual_b)
    new_individual_c = ones(1, 2);
    
    xa = individual_a(:,1);
    ya = individual_a(:,2);
    
    xb = individual_b(:,1);
    yb = individual_b(:,2);
    
    new_individual_c(1) = (xa+xb)/2; 
    new_individual_c(2) = (ya+yb)/2;
    
end

%% Mutate the individual
function new_individual_m = mutate(individual)
    new_individual_m = ones(1, 2);
    
    next_x = individual(:,1);
    next_y = individual(:,2);
    
    
    new_individual_m(1) = next_x + (-0.05+(0.05-(-0.05))*rand(1,1));
    new_individual_m(2) = next_y + (-0.05+(0.05-(-0.05))*rand(1,1));
    
end

%% Generate a new generation
function next_generation = make_next_generation(previous_population, population_size)
    next_generation = ones(population_size, 2);
    sorted_by_fitness_population = sort_population_by_function(previous_population);
    
    % Sum of the results of the individuals
    objective_sum = sum(objective_function(previous_population));
    
    for i=1:population_size
        first_individual = choice_by_roulette(sorted_by_fitness_population, objective_sum);
        second_individual = choice_by_roulette(sorted_by_fitness_population, objective_sum);
        individual = crossover(first_individual, second_individual);
        individual = mutate(individual);
        next_generation(i,:) = individual;
    end
end
