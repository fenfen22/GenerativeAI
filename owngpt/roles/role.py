
"""
The mentor agent can provide context and motivation, 
the coach agent can set clear goals and paths for achievement, 
and the tutor agent can offer detailed explanations and practice opportunities. 

The specific tasks and goals for each agent should align with the learning objectives and the content of the course material provided.
"""

class Role:
    def __init__(self, name="", profile="", goal="", constraints="", desc=""):
        self.llm = LLM()
        # self.


class RoleSetting(BaseModel):
    """Role Settings"""
    name: str
    profile: str
    goal: str
    constraints: str
    desc: str

    def __str__(self):
        return f"{self.name}({self.profile})"

    def __repr__(self):
        return self.__str__()
