"""astra_yam - closed-loop control of the bimanual YAM arms with OpenAI's GPT-6 Astra.

Pipeline:  observation (proprio + cameras)  ->  Astra (Responses API tool call)
           ->  safety gateway (bounds, IK, joint limits, pacing)  ->  robot server (ZMQ)
           ->  new observation  ->  ...  until `done` / `give_up` or a budget/time limit.
"""

__version__ = "0.1.0"
