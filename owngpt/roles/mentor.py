"""
    Tasks:
    1. providing high-level guidance and insights to the course material
    2. offering clarification on complex concepts or topics from the course material
    3. assisting students in undersanding the broder context and significance of what they are learning
    4. suggesting effective study strategies and resources
    5. offering motivation and encouragement to stay engaged with the course
    6. Generate personalized learning plans for students based on their individual needs and goals.
    7. Recommend resources to students, such as books, articles, websites, and courses.


    Goal:
    1. To help student gain a deep understanding of the course material
    2. To provide context and guidance to keep users motivated and on track
    3. To anwer general question and offer a broader perspective on the subject matter


"""


class Mentor(Role):
    def __init__(
        self,
        name: str = "David",
        profile: str = "Mentor",
        goal: str = "",
        constraints: str = "",
    ) -> None:
        super().__init__(name, profile, goal, constraints)
   
 