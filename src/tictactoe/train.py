import env
import agent

agent_1 = agent.Agent(9, 1)
agent_2 = agent.Agent(9, 2)
env = env.Env(agent_1, agent_2)

env.render()
env.play_move(4, agent_1.agent_id)
env.play_move(0, agent_2.agent_id)
print("\nnext move")
env.render()
