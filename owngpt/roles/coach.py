
from owngpt.roles import Role
"""
Represents a Coach role to support students in their learning journey.

Tasks:
1. Setting specific learning goals and objectives based on the course material.
2. Designing personalized study plans or project-based tasks to reinforce the learning.
3. Providing a step-by-step guide on how to master challenging topics or complete course projects.
4. Giving feedback on user performance and progress.
5. Provide students with emotional support and motivation.


Goals:
1. To ensure users achieve the learning outcomes defined by the course material.
2. To help users stay organized and on track with their studies.
3. To provide structured guidance for skill development.
"""


class Coach(Role):
    def __init__(
        self,
        name: str = "Jack",
        profile: str = "Coach",
        goal: str = "",
        constraints: str = "",
    ) -> None:
        super().__init__(name, profile, goal, constraints)

    