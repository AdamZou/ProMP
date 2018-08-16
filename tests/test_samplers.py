import unittest
import numpy as np
from maml_zoo.policies.base import Policy
from maml_zoo.samplers import MAMLSampler
from maml_zoo.samplers import MAMLSampleProcessor
from maml_zoo.samplers import SampleProcessor
from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline

class TestEnv():
    def __init__(self):
        self.state = np.zeros(1)
        self.goal = 0

    def sample_tasks(self, n_tasks):
        """ 
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of reset args
        """
        return np.random.choice(100, n_tasks, replace=False) # Ensure every env has a different goal

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal

    def step(self, action):
        self.state += self.goal - action
        return self.state * 100 + self.goal, (self.goal - action)[0], 0, {'e':self.state}

    def reset(self):
        self.state = np.zeros(1)
        return self.state

    def env_spec(self):
        return None

class RandomEnv(TestEnv):
    def step(self, action):
        self.state += (self.goal - action) * np.random.random()
        return self.state * 100 + self.goal, (self.goal - action)[0], 0, {'e':self.state}

class TestPolicy(Policy):
    def get_actions(self, observations):
        return [[np.ones(1) for batch in task] for task in observations], None

class ReturnPolicy(Policy):
    def get_actions(self, observations):
        return [[batch / 100 for batch in task] for task in observations], None

class RandomPolicy(Policy):
    def get_actions(self, observations):
        return [[np.random.random() * batch for batch in task] for task in observations], [[{'a':1, 'b':2} for batch in task] for task in observations]

class TestSampler(unittest.TestCase):
    def setUp(self):
        self.test_env = TestEnv()
        self.random_env = RandomEnv()
        self.test_policy = TestPolicy(obs_dim=3, action_dim=4)
        self.return_policy = ReturnPolicy(obs_dim=3, action_dim=4)
        self.random_policy = RandomPolicy(obs_dim=3, action_dim=4)
        self.meta_batch_size = 3
        self.batch_size = 4
        self.path_length = 5
        self.it_sampler = MAMLSampler(self.test_env, self.test_policy, self.batch_size, self.meta_batch_size, self.path_length, parallel=False)
        self.par_sampler = MAMLSampler(self.test_env, self.test_policy, self.batch_size, self.meta_batch_size, self.path_length, parallel=True)
        self.sample_processor = SampleProcessor(baseline=LinearFeatureBaseline())
        self.maml_sample_processor = MAMLSampleProcessor(baseline=LinearFeatureBaseline())

    def testSingle(self):
        for sampler in [self.par_sampler]:
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)
            for task in paths.values():
                self.assertEqual(len(task), self.batch_size)
                for path in task:
                    self.assertEqual(len(path), self.path_length)
                    for act in path['actions']:
                        self.assertEqual(act, 1)
                    path_state = 0
                    for obs in path['observations']:
                        self.assertEqual(obs, path_state)
                        path_state += -100

    def testGoalSet(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)

            for task in paths.values(): # All paths in task are equal
                for j in range(self.path_length): # batch size
                    curr_obs = task[0]["observations"][j]
                    for path in task:
                        self.assertEqual(path["observations"][j], curr_obs)
            for j in range(1, self.path_length): # All paths in different tasks are different
                for i in range(self.batch_size):
                    curr_obs = paths[0][i]['observations'][j]
                    for h in range(1, self.meta_batch_size):
                        self.assertNotEqual(paths[h][i]['observations'][j], curr_obs)

    def testRandomSeeds1(self):
        for sampler_parallel in [True, False]:
            np.random.seed(22)
            sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                     self.path_length, parallel=sampler_parallel)
            sampler.update_tasks()
            paths1 = sampler.obtain_samples()

            np.random.seed(22)
            sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                  self.path_length, parallel=sampler_parallel)
            sampler.update_tasks()
            paths2 = sampler.obtain_samples()

            for task1, task2 in zip(paths1.values(), paths2.values()): # All rewards in task are equal, but obs are not
                for j in range(self.batch_size):
                    for k in range(self.path_length):
                        self.assertEqual(task1[j]["observations"][k], task2[j]["observations"][k])

    def testRandomSeeds2(self):
        for sampler_parallel in [True, False]:
            np.random.seed(22)
            sampler = MAMLSampler(self.random_env, self.test_policy, self.batch_size, self.meta_batch_size,
                                  self.path_length, parallel=sampler_parallel)
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)

            for task in paths.values(): # All rewards in task are equal, but obs are not
                for j in range(1, self.path_length): # batch size
                    curr_obs = task[0]["observations"][j]
                    curr_rew = task[0]['rewards'][j]
                    for h in range(1, self.batch_size):
                        self.assertNotEqual(task[h]["observations"][j], curr_obs)
                        self.assertEqual(task[h]['rewards'][j], curr_rew)

    def testInfoDicts(self):
        it_sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                      self.path_length, parallel=False)
        par_sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                       self.path_length, parallel=True)

        for sampler in [it_sampler, par_sampler]:
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)

            for task in paths.values(): # All rewards in task are equal, but obs are not
                for h in range(1, self.batch_size): # batch size
                    curr_agent_infos = task[h]["agent_infos"]
                    curr_env_infos = task[h]['env_infos']
                    self.assertEqual(type(curr_agent_infos), dict)
                    self.assertEqual(type(curr_env_infos), dict)
                    self.assertEqual(len(curr_agent_infos.keys()), 2)
                    self.assertEqual(len(curr_env_infos.keys()), 1)

    def testMAMLSampleProcessor(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.update_tasks()
            paths_meta_batch = sampler.obtain_samples()
            samples_data_meta_batch = self.maml_sample_processor.process_samples(paths_meta_batch)
            self.assertEqual(len(samples_data_meta_batch), self.meta_batch_size)
            for samples_data in samples_data_meta_batch:
                self.assertEqual(len(samples_data.keys()), 7)
                self.assertEqual(samples_data['advantages'].size, self.path_length*self.batch_size)

    def testSampleProcessor(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.update_tasks()
            paths_meta_batch = sampler.obtain_samples()
            for paths in paths_meta_batch.values():
                samples_data = self.sample_processor.process_samples(paths)
                self.assertEqual(len(samples_data.keys()), 7)
                self.assertEqual(samples_data['advantages'].size, self.path_length*self.batch_size)


if __name__ == '__main__':
    unittest.main()