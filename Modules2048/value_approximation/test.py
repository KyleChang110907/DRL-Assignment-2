import torch
import copy
import numpy as np
from student_agent import Game2048Env
from Modules2048.value_approximation.CNN import ValueNetwork, board_to_tensor


def evaluate_value_network(env, value_net, num_episodes=10, gamma=0.99):
    """
    Evaluate the trained CNN value network on the 2048 environment.

    For each episode:
      1. Reset the environment.
      2. At each step, for every legal move:
         a. Create a deep copy of the environment.
         b. Record the current score.
         c. Simulate the move.
         d. Compute the incremental reward as the difference between the new score and the previous score.
         e. Convert the resulting board state to a tensor.
         f. Use the value network to estimate the future value.
         g. Calculate the overall value estimate as: 
                value_est = incremental_reward + gamma * future_value
      3. Choose the action with the highest estimated value and execute it.
      4. Render the board (optional).
      5. Record the final score when the game is over.

    Args:
        env: Instance of the 2048 game environment.
        value_net: The trained CNN value approximator.
        num_episodes: Number of episodes to run for evaluation.
        gamma: Discount factor to weight the future value.

    Returns:
        A tuple containing:
          - A list of final scores for each episode.
          - The average score over all episodes.
          - The standard deviation of scores.
    """

    episode_scores = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            best_action = None
            best_value = -float('inf')
            
            # Evaluate each legal move.
            for action in legal_moves:
                sim_env = copy.deepcopy(env)
                current_score = sim_env.score
                next_state, new_score, done_sim, _ = sim_env.step(action)
                # Compute incremental reward (score gain from the move).
                incremental_reward = new_score - current_score
                # Convert the simulated board to tensor.
                state_tensor = board_to_tensor(sim_env.board)
                with torch.no_grad():
                    future_value = value_net(state_tensor).item()
                # Combine immediate reward with discounted future value.
                value_est = incremental_reward + gamma * future_value
                if value_est > best_value:
                    best_value = value_est
                    best_action = action
            
            # Execute the best action in the real environment.
            state, reward, done, _ = env.step(best_action)
            env.render(action=best_action)
        
        episode_scores.append(env.score)
        print(f"Episode {ep+1}/{num_episodes} finished, final score: {env.score}")
    
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"Average score over {num_episodes} episodes: {avg_score:.2f}")
    print(f"Standard deviation: {std_score:.2f}")
    
    return episode_scores, avg_score, std_score


