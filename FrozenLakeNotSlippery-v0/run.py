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
    max_episode_steps=50,
    reward_threshold=0.78,  # optimum = .8196
)

#######################################################################################################################


class ValueIterationAgent(object):

    """ ... """

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

    def act(self, current_observation, last_action_reward, is_done):

        #print('act() for state='+str(current_observation))

        # Add state, action, reward item to episode history
        if self.last_action != None:
            self.current_episode.append({'action': self.last_action, 'state': self.last_state, 'reward': last_action_reward})

        if self.last_state != None:
            if self.last_state == current_observation:
                # The state did not change, so let's do a random action
                print('Doing random action because state did not change!')
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
    outdir = '/tmp/value-iteration-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = ValueIterationAgent(env.action_space, env.observation_space)
    
    episode_count = 4000
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
                # Do last action computation. Required in order to apply the
                # last reward to the episode history before learning from it.
                action = agent.act(current_observation, last_reward, done)
                print('Used steps = '+str(steps))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        agent.learn_from_episode()

        print('END ---------------')

    print(agent.g)
    print(agent.final_rewards)

    # Close the env and write monitor result info to disk
    env.close()
