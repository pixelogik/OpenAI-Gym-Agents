import argparse
import sys
import gym

import numpy as np
from gym import wrappers, logger
from gym.envs.registration import register

#######################################################################################################################

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': True},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

#######################################################################################################################

class NaiveAgent(object):

    """ 
    An agent I wrote without knowing what I am doing. Works well for deterministic
    environments (is_slippery = False), does not work for non-deterministic environments. 
    """

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.A = action_space.n
        self.S = observation_space.n

        # For analytics purposes we store the final reward for each episode in here
        self.final_rewards = []

        # Create array of gain history for all (state,action) pairs
        self.q = np.empty((self.S, self.A), dtype=object)

        # Create array of estimated gain for all (state,action) pairs
        self.g = np.empty((self.S, self.A))

        # Initialize gain history for every pair with one 0.5 entry
        # this way the failing paths with reward 0.0 are less interesting
        # as unexplored paths. This way the agent explores the unknown.
        for s in range(self.S):
            for a in range(self.A):
                self.q[s][a] = [0.5]
                self.g[s][a] = 0.0

        # In this array we will store the (state, action, reward) items in
        # order to adapt the value function after each episode
        self.current_episode = []

        # We will cache last action and state in here in order to add them
        # to the episode history once the reward is there
        self.last_action = None
        self.last_state = None

    def start_episode(self):
        self.current_episode = []
        self.last_action = None
        self.last_state = None

    def reset_evaluation(self):
        self.final_rewards = []

    def evaluate_episode(self):
        final_reward = self.current_episode[-1]['reward']
        print('Final reward: ' + str(final_reward))
        self.final_rewards.append(final_reward)

    def learn_from_episode(self):
        #print('Current episode:')
        # print(self.current_episode)

        final_reward = self.current_episode[-1]['reward']
        print('Final reward: ' + str(final_reward))

        self.final_rewards.append(final_reward)

        blend_factor = 1.0

        for step in reversed(self.current_episode):
            mixed_reward = (1.0-blend_factor)*step['reward'] + final_reward*blend_factor
            self.q[step['state']][step['action']].append(mixed_reward)

            blend_factor = blend_factor * 0.9

            if blend_factor <= 0.01:
                blend_factor = 0.01

        # print('Q(s,a):')
        # print(self.q)

    def finalize_episode(self, current_observation, last_action_reward):
        if self.last_action != None:
            self.current_episode.append({'action': self.last_action, 'state': self.last_state, 'reward': last_action_reward})

    def act(self, current_observation, last_action_reward, is_done):

        #print('act() for state='+str(current_observation))

        # Add state, action, reward item to episode history
        if self.last_action != None:
            self.current_episode.append({'action': self.last_action, 'state': self.last_state, 'reward': last_action_reward})

        # This actually performs worse!!!
        #
        # if self.last_state != None:
        #     if self.last_state == current_observation:
        #         # The state did not change, so let's do a random action
        #         print('Doing random action because state did not change!')
        #         random_action = self.action_space.sample()

        #         # Remember state and action for when reward is coming in and
        #         # we add it all to the episode history
        #         self.last_action = random_action
        #         self.last_state = current_observation

        #         return random_action

        action_candidates = []

        # Check estimated gain for every action based on history
        for possible_action in range(self.A):

            #print('Possible action '+str(possible_action))

            estimated_gain = 0.0
            past_gains = self.q[current_observation][possible_action]

            # print(past_gains)

            # Compute estimated gain for this action based on history
            for past_gain in past_gains:
                estimated_gain = estimated_gain + past_gain

            #print('len(past_gains) = '+str(len(past_gains)))
            estimated_gain = estimated_gain / len(past_gains)

            # Cache estimated gain for debugging purposes
            self.g[current_observation][possible_action] = estimated_gain

            # Append action as candidate with estimated gain
            action_candidates.append({'gain': estimated_gain, 'action': possible_action})

        # Sort with respect to estimated gain
        sorted_action_candidates = sorted(action_candidates, key=lambda x: x['gain'], reverse=True)

        #print('Sorted action candidates:')
        # print(sorted_action_candidates)

        # Remember state and action for when reward is coming in and
        # we add it all to the episode history
        self.last_action = sorted_action_candidates[0]['action']
        self.last_state = current_observation

        # Return action with largest estimated gain
        return self.last_action

#######################################################################################################################

#######################################################################################################################

class QLearningAgent(object):

    """ 
    Q learning agent 
    """

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.A = action_space.n
        self.S = observation_space.n

        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 1.0

        # We do epsilon greedy exploration. In the beginning the agent will act more 
        # randomly in order to explore the state / action space. Later on it will act 
        # more accordingly to the value function for (state,action) pairs
        self.epsilon = 0.9
        self.epsilon_decay = 0.99

        # For analytics purposes we store the final reward for each episode in here
        self.final_rewards = []

        # Create array of Q for all (state,action) pairs
        self.q = np.empty((self.S, self.A))

        # Initialize Q for every pair with one 0.5 entry
        # this way the failing paths with reward 0.0 are less interesting
        # as unexplored paths. This way the agent explores the unknown.
        for s in range(self.S):
            for a in range(self.A):
                self.q[s][a] = 0.5

        # In this array we will store the (state, action, reward) items in
        # order to adapt the value function after each episode
        self.current_episode = []

        # We will cache last action and state in here in order to add them
        # to the episode history once the reward is there
        self.last_action = None
        self.last_state = None

    def reset_evaluation(self):
        self.final_rewards = []
        self.epsilon = 0.0

    def evaluate_episode(self):
        final_reward = self.current_episode[-1]['reward']
        print('Final reward: ' + str(final_reward))
        self.final_rewards.append(final_reward)

    def start_episode(self):
        self.current_episode = []
        self.last_action = None
        self.last_state = None

        # Decrease greedy exploration parameter epsilon
        self.epsilon = self.epsilon * self.epsilon_decay

    def learn_from_episode(self):
        #print('Current episode:')
        # print(self.current_episode)

        final_reward = self.current_episode[-1]['reward']
        print('Final reward: ' + str(final_reward))

        self.final_rewards.append(final_reward)

        lr = self.learning_rate
        df = self.discount_factor

        # Compute maxium future reward from next step
        optimal_future_reward = 0.0

        next_state = None
        
        for step in reversed(self.current_episode):

            if next_state == None:
                # Last state of episode
                learned_value = step['reward']
                self.q[step['state']][step['action']] = (1.0-lr) * self.q[step['state']][step['action']] + lr * learned_value
            else:
                # Not last state of episode
                optimal_future_reward = 0.0

                # Get maximum future reward for (s,a) from next step in episode
                for possible_action in range(self.A):
                    future_action_reward = self.q[next_state][possible_action]
                    if future_action_reward > optimal_future_reward:
                        optimal_future_reward = future_action_reward

                learned_value = step['reward'] + df * optimal_future_reward
                self.q[step['state']][step['action']] = (1.0-lr) * self.q[step['state']][step['action']] + lr * learned_value

            next_state = step['state']

        # print(self.q)

    def finalize_episode(self, current_observation, last_action_reward):
        if self.last_action != None:
            self.current_episode.append({'action': self.last_action, 'state': self.last_state, 'reward': last_action_reward})

    def act(self, current_observation, last_action_reward, is_done):

        #print('act() for state='+str(current_observation))

        # Add state, action, reward item to episode history
        if self.last_action != None:
            self.current_episode.append({'action': self.last_action, 'state': self.last_state, 'reward': last_action_reward})

        # Do random action if greedy exploration parameter is higher then random number
        if np.random.rand() < self.epsilon:
            print('Doing random action because greedy!')
            random_action = self.action_space.sample()

            # Remember state and action for when reward is coming in and
            # we add it all to the episode history
            self.last_action = random_action
            self.last_state = current_observation

            return random_action

        action_candidates = []

        # Check estimated gain for every action based on history
        for possible_action in range(self.A):

            #print('Possible action '+str(possible_action))
            estimated_gain = self.q[current_observation][possible_action]

            # Append action as candidate with estimated gain
            action_candidates.append({'gain': estimated_gain, 'action': possible_action})

        # Sort with respect to estimated gain
        sorted_action_candidates = sorted(action_candidates, key=lambda x: x['gain'], reverse=True)

        #print('Sorted action candidates:')
        # print(sorted_action_candidates)

        # Remember state and action for when reward is coming in and
        # we add it all to the episode history
        self.last_action = sorted_action_candidates[0]['action']
        self.last_state = current_observation

        # Return action with largest estimated gain
        return self.last_action

#######################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='FrozenLakeNotSlippery-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/frozen-lake-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QLearningAgent(env.action_space, env.observation_space)
    
    episode_count = 2000
    last_reward = 0
    action = 0
    done = False

    for i in range(episode_count):
        # print("#########################################")
        current_observation = env.reset()

        print('BEGIN ---------------')
        agent.start_episode()

        steps = 0

        while True:
            # for k in range(3):
            # print('---------------------------------------')
            action = agent.act(current_observation, last_reward, done)
            current_observation, last_reward, done, p = env.step(action)

            steps = steps + 1

            #print('Action: '+str(action))
            #print('Reward: '+str(last_reward))
            #print('New observation: '+str(current_observation))
            #print('Done: '+str(done))
            #print('P: '+str(p))
            if done:
                env.render()
                agent.finalize_episode(current_observation, last_reward)
                print('Used steps = '+str(steps))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        agent.learn_from_episode()

        print('END ---------------')

    # Prepare agent for evaluation
    agent.reset_evaluation()

    for i in range(100):
        current_observation = env.reset()
        agent.start_episode()
        steps = 0
        while True:
            action = agent.act(current_observation, last_reward, done)
            current_observation, last_reward, done, p = env.step(action)
            steps = steps + 1

            if done:
                env.render()
                agent.finalize_episode(current_observation, last_reward)
                agent.evaluate_episode()
                print('Used steps = '+str(steps))
                break

        print('END ---------------')

    last_100_rewards = agent.final_rewards
    print(last_100_rewards)
    print('SUCCESS RATE = '+str(100.0*np.sum(last_100_rewards) / len(last_100_rewards))+'%')

    # Close the env and write monitor result info to disk
    env.close()
