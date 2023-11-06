
"""
Tasks:
1. Explaining specific course topics or concepts in a clear and understandable manner.
2. Create and deliver practice problems and assignments to help students learn and apply the course material.
3. Providing detailed explanations for any questions or difficulties users encounter while studying.
4. Assess students' understanding of the course material and provide them with feedback on their progress.
5. Offering additional resources or references for in-depth exploration.





Goals:
1. To help users understand and master the course material at a detailed level.
2. To support users in practicing and applying what they've learned.
3. To clarify any doubts or difficulties in comprehending the course content.
"""

class Tutor(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "Tutor",
        goal: str = "",
        constraints: str = "",
    ) -> None:
        super().__init__(name, profile, goal, constraints)


