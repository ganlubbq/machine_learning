clear all; close all; clc;
fig = 0;  % Counter for labeling figures

A_titles = {'diag(1,1)', 'diag(2,1)', 'diag(20,1)', ...
			'V\Lambda_1V^{-1}', 'V\Lambda_2V^{-1}'};

% Loops through the problems with different A matrices

for problem = 1:5

	%% Setup
	% Defines all possible w vectors within range
	max = 20;
	min = -20;
	int = 1;
	[w1, w2] = meshgrid(min:int:max);

	% Given values
	wo = [-2, 2]';
	step = [0.09, 5, 0.5];
	wo_init = [5, 15]';
	err_min = 0.001;
	err_max = 300;
	iterations = 0;

	% Changes A matrix values based on current problem
	switch problem
	case 1
		A = diag([1, 1]);
	case 2
		A = diag([2, 1]);
	case 3
		A = diag([20, 1]);
	case 4
        V = [cos(pi/4), sin(pi/4); -sin(pi/4), cos(pi/4)];
        L = diag([2, 1]);
		A = V * L * V^-1;
	case 5
        V = [cos(pi/4), sin(pi/4); -sin(pi/4), cos(pi/4)];
        L = diag([20, 1]);
		A = V * L * V^-1;
	end

	% Calculates the cost function
	jw = zeros(length(w1), length(w2));
	for i = 1:length(w1)
		for j = 1:length(w2)
			w  = ([w1(i,j), w2(i,j)]');
			jw(i,j) = (w - wo)' * A * (w - wo);
		end
	end

	%% A - Plots contour and surface
	figure, fig = fig + 1;
	contour(w1, w2, jw, 'ShowText','on'); axis equal;
	xlabel('w_1'); ylabel('w_2');
	title(['Fig ' num2str(fig) ' - Contour plot of J(w), A=' A_titles{problem}]);
	print(['fig' num2str(fig)],'-djpeg', '-r0');

	figure, fig = fig + 1;
	surf(w1, w2, jw);
	xlabel('w_1'); ylabel('w_2'); zlabel('z');
	title(['Fig ' num2str(fig) ' - Surface plot of J(w), A=' A_titles{problem}]);
	print(['fig' num2str(fig)],'-djpeg', '-r0');

	%% B - Calculates gradient and plots quiver
	[x, y] = gradient(jw);
	figure, fig = fig + 1;
	quiver(w1, w2, x, y); axis equal;
	xlabel('w_1'); ylabel('w_2');
	axis([min max min max])
	title(['Fig ' num2str(fig) ' - Quiver plot of J(w), A=' A_titles{problem}]);
	print(['fig' num2str(fig)],'-djpeg', '-r0');

	%% C, D - Runs gradient descent algorithm with different step sizes

	% Gradient descent for given step sizes
	for i = 1:2
		% Start fresh
		iterations = 0;
		w = wo_init';
		k = 1;
		err = (w(k, :)' - wo) .^ 2;

		% Gradient descent algorithm to find minimum
		while 1
            grad = A * (w(k, :)' - wo);
            w(k+1, :) = (w(k, :)' - step(i) * grad)';
            err = (w(k+1, 1) - wo(1)) .^ 2 + (w(k+1, 2) - wo(2)) .^ 2;
			k = k + 1;
			iterations = iterations + 1;
			if (err < err_min) || (iterations > err_max)
				break
			end
        end

    PROBLEM = problem
    STEP = i
    GRAD = w(k,:)
	ITERATIONS = iterations

		figure, fig = fig + 1;
		contour(w1, w2, jw);  axis equal; hold on;
		plot(w(:,1), w(:,2), '-o'); hold off;
		axis([min max min max])
		title(['Fig ' num2str(fig) ' - Gradient descent of J(w), A=' A_titles{problem} ', \mu=' num2str(step(i))]);
		print(['fig' num2str(fig)],'-djpeg', '-r0');
    end
    
    for i = 3
		% Start fresh
		iterations = 0;
		w = wo_init';
		k = 1;
		err = (w(k, :)' - wo) .^ 2;
        j = 1;

		% Gradient descent algorithm to find minimum
		while 1
            % Line search
			grad = A * (w(k, :)' - wo);
            w(k+1, :) = (w(k, :)' - step(i) * grad)';
            min_jw(j+1) = (w(k+1, :)' - wo)' * A * (w(k+1, :)' - wo);
            if (min_jw(j+1) - min_jw(j)) <= err_min;
                step(i) = 1;
                break
            else
                step(i) = step(i) / 2;
                j = j + 1;
            end
            err = (w(k+1, 1) - wo(1)) .^ 2 + (w(k+1, 2) - wo(2)) .^ 2;
			k = k + 1;
			iterations = iterations + 1;
			if (err < err_min) || (iterations > err_max)
				break
			end
        end

    PROBLEM = problem
    STEP = i
    	LINE = w(k,:)
	ITERATIONS = iterations
        
        figure, fig = fig + 1;
		contour(w1, w2, jw);  axis equal; hold on;
		plot(w(:,1), w(:,2), '-o'); hold off;
		axis([min max min max])
		title(['Fig ' num2str(fig) ' - Gradient descent of J(w), A=' A_titles{problem} ', \mu=line search']);
		print(['fig' num2str(fig)],'-djpeg', '-r0');
    end

	%% E - Newton's Method
	% Calculates the cost function
	fw = zeros(length(w1), length(w2));
	for i = 1:length(w1)
		for j = 1:length(w2)
			w  = ([w1(i,j), w2(i,j)]');
			fw(i,j) = (w - wo)' * (1/2) * 2 * A * (w - wo);
		end
    end
    
	% Calculates gradient and plots quiver
	[x, y] = gradient(fw);
	figure, fig = fig + 1;
	quiver(w1, w2, x, y); axis equal;
	xlabel('w_1'); ylabel('w_2');
	axis([min max min max])
	title(['Fig ' num2str(fig) ' - Quiver plot of F(w), A=' A_titles{problem}]);
	print(['fig' num2str(fig)],'-djpeg', '-r0');
    

	%% F - Newton's Method algorithm
	% Start fresh
	iterations = 0;
	w = wo_init';
	k = 1;
	err = (w(k, :)' - wo) .^ 2;

	while 1
		newt = -(w(k, :)' - wo);
		w(k+1, :) = (w(k, :)' + newt)';
		err = (w(k+1, 1) - wo(1)) .^ 2 + (w(k+1, 2) - wo(2)) .^ 2;
		k = k + 1;
		iterations = iterations + 1;
		if (err < err_min) || (iterations > err_max)
			break
		end
    end

    PROBLEM = problem
	NEWTON = w(k,:)
	ITERATIONS = iterations

	figure, fig = fig + 1;
	contour(w1, w2, jw);  axis equal; hold on;
	plot(w(:,1), w(:,2), '-o'); hold off;
	axis([min max min max])
	title(['Fig ' num2str(fig) ' - Newtons Method for F(w), A=' A_titles{problem}]);
	print(['fig' num2str(fig)],'-djpeg', '-r0');
	
end

