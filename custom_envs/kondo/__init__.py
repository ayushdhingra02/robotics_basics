from gymnasium.envs.registration import register

register(id='Kondo-v0',
         entry_point='custom_envs.kondo.envs:HumanoidEnv')