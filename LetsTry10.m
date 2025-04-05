% Setup figure for 2D map with obstacles, target, and moving dot
figure('Name', '2D Map with Obstacles and Lidar Localization');
hold on;
axis([0 10 0 10]);
grid on;

% Add labels with units
xlabel('X Position (metres)');
ylabel('Y Position (metres)');

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
waypoints = [1, 1; 1, 4; 5, 4; 5, 9; 9, 9];
steps = 500;
motion_error_std = 0.1; % Slight noise for motion

% Particle filter parameters
num_particles = 500;
lidar_max_range = 1;
lidar_noise_std = 0.5;

% Initialize robot position, particles, and weights
robot_pos = [1, 1]; % Starting position
particles = [rand(num_particles, 1) * 10, rand(num_particles, 1) * 10]; % Global distribution
weights = ones(num_particles, 1) / num_particles;

% Visualization handles
robot_plot = plot(robot_pos(1), robot_pos(2), 'bo', 'MarkerFaceColor', 'b', ...
                  'MarkerSize', 12, 'DisplayName', 'Robot'); % Larger size for the blue dot
particles_plot = scatter(particles(:, 1), particles(:, 2), 10, weights, 'MarkerEdgeColor', 'g', ...
                         'MarkerFaceColor', 'none', 'DisplayName', 'Particles'); % Hollow particles
colormap('autumn'); % Colormap for weight-based fading
estimated_plot = plot(NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Estimated Position');
legend;

% Main simulation loop
while true
    % Move the robot
    robot_pos = move_robot_straight(robot_pos, waypoints, motion_error_std);
    
    % Check if robot has reached the final waypoint
    if all(abs(robot_pos - waypoints(end, :)) < 0.2) % Final position threshold
        disp('Robot reached the target location.');
        break;
    end

    % Get real lidar measurement
    real_lidar = get_lidar_measurements(robot_pos, lidar_max_range, lidar_noise_std);
    
    % Move particles mimicking the robot's motion
    particles = move_particles_mimic_robot(particles, robot_pos, motion_error_std);
    
    % Predict lidar measurements for particles
    predicted_lidar = arrayfun(@(i) get_lidar_measurements(particles(i, :), lidar_max_range, 0), ...
                               1:num_particles, 'UniformOutput', false);
    predicted_lidar = cell2mat(predicted_lidar');

    % Calculate weights based on lidar measurement errors
    errors = sum((predicted_lidar - real_lidar).^2, 2);
    weights = exp(-errors / (2 * lidar_noise_std^2));
    weights = weights / (sum(weights) + eps);
    
    % Estimate position
    estimated_pos = sum(particles .* weights, 1);
    
    % Resample particles
    edges = cumsum(weights);
    edges(end) = 1; % Ensure numerical stability
    u = rand / num_particles;
    indices = zeros(num_particles, 1);
    for i = 1:num_particles
        indices(i) = find(u < edges, 1);
        u = u + 1 / num_particles;
    end
    particles = particles(indices, :);

    % Visualization updates
    set(robot_plot, 'XData', robot_pos(1), 'YData', robot_pos(2));
    set(particles_plot, 'XData', particles(:, 1), 'YData', particles(:, 2), 'CData', weights); % Update weights
    set(estimated_plot, 'XData', estimated_pos(1), 'YData', estimated_pos(2));
    drawnow;
end

% Hide particles in the final display
set(particles_plot, 'Visible', 'off');

% Display final robot and estimated positions
disp(['Final Robot Position: ', mat2str(robot_pos)]);
disp(['Final Estimated Position: ', mat2str(estimated_pos)]);

disp('Simulation complete.');

function new_pos = move_robot_straight(current_pos, waypoints, motion_error_std)
    persistent current_waypoint_index;
    if isempty(current_waypoint_index)
        current_waypoint_index = 1; % Start at the first waypoint
    end

    % Check if we have reached the last waypoint
    if current_waypoint_index > size(waypoints, 1)
        new_pos = current_pos;
        return; % Stay at the final position
    end

    % Get the current target waypoint
    target_waypoint = waypoints(current_waypoint_index, :);

    % Calculate the direction vector toward the current waypoint
    direction = target_waypoint - current_pos;
    distance_to_waypoint = norm(direction);

    % If close enough to the current waypoint, move to the next waypoint
    if distance_to_waypoint < 0.2 % Threshold for waypoint switching
        current_waypoint_index = current_waypoint_index + 1;

        % Check if we are at the final waypoint
        if current_waypoint_index > size(waypoints, 1)
            new_pos = target_waypoint; % Snap to the final position
            return;
        end

        % Update the target waypoint
        target_waypoint = waypoints(current_waypoint_index, :);
        direction = target_waypoint - current_pos;
        distance_to_waypoint = norm(direction);
    end

    % Normalize the direction vector and move a fixed step size
    direction = direction / max(distance_to_waypoint, eps); % Avoid division by zero
    step_size = min(0.2, distance_to_waypoint); % Step size limited by remaining distance
    new_pos = current_pos + step_size * direction;

    % Add minimal noise to simulate real-world motion (optional)
    new_pos = new_pos + motion_error_std * randn(1, 2);

    % Keep the robot within map bounds (optional safety check)
    new_pos = max(new_pos, [0.5, 0.5]); % Minimum boundary
    new_pos = min(new_pos, [9.5, 9.5]); % Maximum boundary
end

% Function to move particles mimicking the robot
function new_particles = move_particles_mimic_robot(particles, robot_pos, motion_error_std)
    % Move particles the same way the robot moves but starting from random positions
    displacement = (robot_pos - mean(particles, 1));
    new_particles = particles + displacement + motion_error_std * randn(size(particles));
end

% Function to get lidar measurements
function lidar_readings = get_lidar_measurements(pos, max_range, noise_std)
    directions = [1, 0; 0, 1; -1, 0; 0, -1]; % N, E, S, W
    lidar_readings = max_range * ones(1, size(directions, 1));

    for i = 1:size(directions, 1)
        for d = 0:0.1:max_range
            test_point = pos + d * directions(i, :);
            if test_point(1) < 0 || test_point(1) > 10 || test_point(2) < 0 || test_point(2) > 10 || ...
                    ((test_point(1) > 2 && test_point(1) < 4 && test_point(2) > 2 && test_point(2) < 3) || ...
                    (test_point(1) > 6 && test_point(1) < 7 && test_point(2) > 5 && test_point(2) < 8))
                lidar_readings(i) = d + noise_std * randn();
                break;
            end
        end
    end
end