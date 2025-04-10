% Main Script to Compare Resampling Techniques

% Define resampling methods to compare
methods = {'multinomial', 'systematic', 'residual', 'stratified'};
num_methods = length(methods);
errors_all = cell(num_methods, 1);

% Run simulation for each resampling method
for m = 1:num_methods
    method = methods{m};
    disp(['Running simulation with ', method, ' resampling']);
    errors = run_simulation(method);
    errors_all{m} = errors;
end

% Plot the errors
figure;
hold on;
colors = lines(num_methods);
for m = 1:num_methods
    plot(errors_all{m}, 'Color', colors(m,:), 'DisplayName', methods{m});
end
hold off;
legend('show');
xlabel('Time Step');
ylabel('Localization Error');
title('Localization Error Over Time for Different Resampling Techniques');
grid on;

% Simulation Function
function errors = run_simulation(method)
    % Setup figure for 2D map with complex obstacles, target, and particle filter
    figure('Name', ['Simulation with ', method, ' Resampling']);
    hold on;
    axis([0 10 0 10]);
    grid on;

    % Obstacles
    rectangle('Position', [2, 2, 2, 1], 'FaceColor', 'k'); % Obstacle 1
    rectangle('Position', [6, 5, 1, 3], 'FaceColor', 'k'); % Obstacle 2

    % Boundary walls
    rectangle('Position', [0, 9.5, 10, 0.5], 'FaceColor', 'k'); % Top
    rectangle('Position', [0, 0, 10, 0.5], 'FaceColor', 'k'); % Bottom
    rectangle('Position', [0, 0, 0.5, 10], 'FaceColor', 'k'); % Left
    rectangle('Position', [9.5, 0, 0.5, 10], 'FaceColor', 'k'); % Right

    % Target position
    plot(9, 9, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    % Define waypoints for movement
    waypoints = [1, 1; 1, 4; 5, 4; 5, 6; 6, 6; 6, 9; 9, 9];
    motion_error_std = 0.01;

    % Particle filter parameters
    num_particles = 500;
    lidar_max_range = 0.7;
    lidar_noise_std = 0.2;

    % Initialize robot position, particles, weights, and waypoint index
    robot_pos = [1, 1];
    particles = [rand(num_particles, 1) * 10, rand(num_particles, 1) * 10];
    weights = ones(num_particles, 1) / num_particles;
    current_waypoint_index = 1; % Initialize waypoint index here

    % Visualization handles
    robot_plot = plot(robot_pos(1), robot_pos(2), 'bo', 'MarkerFaceColor', 'b', ...
        'MarkerSize', 12, 'DisplayName', 'Robot');
    particles_plot = scatter(particles(:, 1), particles(:, 2), 10, weights, 'MarkerEdgeColor', 'g', ...
        'MarkerFaceColor', 'none', 'DisplayName', 'Particles');
    colormap('jet');
    estimated_plot = plot(NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Estimated Position');
    legend('show');

    % Add Stop/Resume Button
    global isPaused;
    isPaused = false;
    stopButton = uicontrol('Style', 'togglebutton', 'String', 'Pause', ...
        'Position', [10, 10, 80, 30], 'Callback', @togglePause);

    % Threshold ring
    theta = linspace(0, 2*pi, 100);
    circle_x = cos(theta);
    circle_y = sin(theta);
    circle_plot = plot(NaN, NaN, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1, 'LineStyle', '--', ...
        'DisplayName', 'Threshold Area');

    % Set random seed for consistency across runs
    rng(0);

    % Main simulation loop
    errors = [];
    step_count = 0;

    while true
        pause(0.05);
        while isPaused
            pause(0.1);
        end

        % Move the robot and update waypoint index
        prev_robot_pos = robot_pos;
        [robot_pos, current_waypoint_index] = move_robot_straight(robot_pos, waypoints, motion_error_std, current_waypoint_index);

        % Check if robot has reached the final waypoint
        if all(abs(robot_pos - waypoints(end, :)) < 0.2)
            disp('Robot reached the target location.');
            break;
        end

        % Get real lidar measurement
        real_lidar = get_lidar_measurements(robot_pos, lidar_max_range, lidar_noise_std);

        % Calculate displacement
        displacement = robot_pos - prev_robot_pos;

        % Move particles
        particles = move_particles_mimic_robot(particles, displacement, motion_error_std);

        % Predict lidar measurements for particles
        predicted_lidar = arrayfun(@(i) get_lidar_measurements(particles(i, :), lidar_max_range, 0), ...
            1:num_particles, 'UniformOutput', false);
        predicted_lidar = cell2mat(predicted_lidar');

        % Calculate weights
        errors_lidar = sum((predicted_lidar - real_lidar).^2, 2);
        weights = gaussian_pdf(errors_lidar, 0, lidar_noise_std);
        weights = weights / (sum(weights) + eps);

        % Estimate position
        estimated_pos = sum(particles .* weights, 1);

        % Resample particles using specified method
        particles = resample_particles(particles, weights, method);
        weights = ones(num_particles, 1) / num_particles; % Reset weights

        % Compute standard deviation for uncertainty
        radius = mean(std(particles, 0, 1));

        % Update threshold ring
        set(circle_plot, 'XData', estimated_pos(1) + radius * circle_x, ...
            'YData', estimated_pos(2) + radius * circle_y);

        % Visualization updates
        set(robot_plot, 'XData', robot_pos(1), 'YData', robot_pos(2));
        set(particles_plot, 'XData', particles(:, 1), 'YData', particles(:, 2), ...
            'CData', weights, 'SizeData', 20);
        set(estimated_plot, 'XData', estimated_pos(1), 'YData', estimated_pos(2));

        % Track error
        step_count = step_count + 1;
        errors(step_count) = norm(robot_pos - estimated_pos);

        drawnow;
    end

    % Hide particles in final display
    set(particles_plot, 'Visible', 'off');
end

% Modified Motion Model for Robot
function [new_pos, current_waypoint_index] = move_robot_straight(current_pos, waypoints, motion_error_std, current_waypoint_index)
    if current_waypoint_index > size(waypoints, 1)
        new_pos = current_pos;
        return;
    end

    target_waypoint = waypoints(current_waypoint_index, :);
    direction = target_waypoint - current_pos;
    distance_to_waypoint = norm(direction);
    if distance_to_waypoint < 0.2
        current_waypoint_index = current_waypoint_index + 1;
        if current_waypoint_index > size(waypoints, 1)
            new_pos = target_waypoint;
            return;
        end
        target_waypoint = waypoints(current_waypoint_index, :);
        direction = target_waypoint - current_pos;
        distance_to_waypoint = norm(direction);
    end
    direction = direction / max(distance_to_waypoint, eps);
    step_size = min(0.2, distance_to_waypoint);
    new_pos = current_pos + step_size * direction;
    new_pos = new_pos + motion_error_std * randn(1, 2);
    new_pos = max(new_pos, [0.5, 0.5]);
    new_pos = min(new_pos, [9.5, 9.5]);
end

% Motion Model for Particles
function new_particles = move_particles_mimic_robot(particles, displacement, motion_error_std)
    new_particles = particles + displacement + motion_error_std * randn(size(particles));
    new_particles = max(new_particles, [0.5, 0.5]);
    new_particles = min(new_particles, [9.5, 9.5]);
end

% Enhanced LIDAR Measurement Function
function lidar_readings = get_lidar_measurements(pos, max_range, noise_std)
    directions = [1, 0; 0, 1; -1, 0; 0, -1];
    lidar_readings = max_range * ones(1, size(directions, 1));
    for i = 1:size(directions, 1)
        for d = 0:0.01:max_range
            test_point = pos + d * directions(i, :);
            if test_point(1) <= 0.5 || test_point(1) >= 9.5 || test_point(2) <= 0.5 || test_point(2) >= 9.5 || ...
               (test_point(1) > 2 && test_point(1) < 4 && test_point(2) > 2 && test_point(2) < 3) || ... % Obstacle 1
               (test_point(1) > 6 && test_point(1) < 7 && test_point(2) > 5 && test_point(2) < 8) % Obstacle 2
                lidar_readings(i) = d + noise_std * randn();
                break;
            end
        end
    end
end

% Custom Gaussian PDF Function
function y = gaussian_pdf(x, mu, sigma)
    y = (1/(sigma*sqrt(2*pi))) * exp(-((x-mu).^2)/(2*sigma^2));
end

% Resampling Functions
function particles = resample_particles(particles, weights, method)
    num_particles = size(particles, 1);
    switch method
        case 'multinomial'
            indices = multinomial_resample(weights, num_particles);
        case 'systematic'
            indices = systematic_resample(weights, num_particles);
        case 'residual'
            indices = residual_resample(weights, num_particles);
        case 'stratified'
            indices = stratified_resample(weights, num_particles);
        otherwise
            error('Unknown resampling method');
    end
    particles = particles(indices, :);
end

function indices = multinomial_resample(weights, num_particles)
    cum_weights = cumsum(weights);
    cum_weights(end) = 1; % Ensure numerical stability
    u = rand(num_particles, 1);
    indices = zeros(num_particles, 1);
    for i = 1:num_particles
        indices(i) = find(u(i) < cum_weights, 1);
    end
end

function indices = systematic_resample(weights, num_particles)
    edges = cumsum(weights);
    edges(end) = 1; % Ensure numerical stability
    u = rand / num_particles;
    indices = zeros(num_particles, 1);
    for i = 1:num_particles
        indices(i) = find(u < edges, 1);
        u = u + 1 / num_particles;
        if u >= 1
            u = u - 1;
        end
    end
end

function indices = residual_resample(weights, num_particles)
    M = length(weights);
    N = num_particles;
    w_scaled = N * weights;
    int_part = floor(w_scaled);
    total_int = sum(int_part);
    R = N - total_int;
    if R > 0
        residual = w_scaled - int_part;
        if sum(residual) > 0
            residual = residual / sum(residual); % Normalize
            indices_residual = multinomial_resample(residual, R);
        else
            indices_residual = randi(M, R, 1); % Fallback if residuals are zero
        end
    else
        indices_residual = [];
    end
    indices = [];
    for m = 1:M
        indices = [indices; repmat(m, int_part(m), 1)];
    end
    indices = [indices; indices_residual];
    indices = indices(randperm(N)); % Shuffle
end

function indices = stratified_resample(weights, num_particles)
    cum_weights = cumsum(weights);
    cum_weights(end) = 1; % Ensure numerical stability
    u = ((0:num_particles-1)' + rand(num_particles, 1)) / num_particles;
    indices = zeros(num_particles, 1);
    for i = 1:num_particles
        indices(i) = find(u(i) < cum_weights, 1);
    end
end

% Pause/Resume Function
function togglePause(src, ~)
    global isPaused;
    isPaused = get(src, 'Value');
    if isPaused
        set(src, 'String', 'Resume');
    else
        set(src, 'String', 'Pause');
    end
end