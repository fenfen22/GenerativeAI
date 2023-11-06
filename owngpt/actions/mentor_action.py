from owngpt.actions import Action

class MentorAction(Action):
    def __init__(self, name, context=Nonw, llm= None):
        super().__init__(name, context, llm)
        self.desc = "Provide guidance and support to students"
    
    async def provide_high_level_guidance_and_insights(self, course_material):
        """Provides a high-level overview of the course material, highlighting the key concepts and topics, explaining the relationships between different concepts and topics, and discussing the implications of the course material for the real world."""

        # Prompt the LLM to provide a high-level overview of the course material, highlighting the key concepts and topics, explaining the relationships between different concepts and topics, and discussing the implications of the course material for the real world.
        prompt = (
            "Provide a high-level overview of the following course material, highlighting the key concepts and topics, explaining the relationships between different concepts and topics, and discussing the implications of the course material for the real world:\n\n"
            + course_material
        )

        # Send the prompt to the LLM and get the response.
        response = await self._aask(prompt)

        # Return the response.
        return response

    async def offer_clarification_on_complex_concepts_or_topics(self, course_material, complex_concept_or_topic):
        """Provides clarification on a complex concept or topic from the course material."""

        # Prompt the LLM to provide clarification on the complex concept or topic.
        prompt = (
            "Provide clarification on the following complex concept or topic from the course material:\n\n"
            + course_material + "\n\n"
            + complex_concept_or_topic
        )

        # Send the prompt to the LLM and get the response.
        response = await self._aask(prompt)

        # Return the response.
        return response

    async def assist_students_in_understanding_the_broader_context_and_significance_of_what_they_are_learning(self, course_material):
        """Assists students in understanding the broader context and significance of what they are learning."""

        # Prompt the LLM to discuss the historical background of the course material, the current state of the art in the field, and the potential impact of the course material on society.
        prompt = (
            "Discuss the historical background, the current state of the art in the field, and the potential impact on society of the following course material:\n\n"
            + course_material
        )

        # Send the prompt to the LLM and get the response.
        response = await self._aask(prompt)

        # Return the response.
        return response

    async def suggest_effective_study_strategies_and_resources(self, student_learning_style, course_material):
        """Suggests effective study strategies and resources based on the student's learning style."""

        # Prompt the LLM to suggest effective study strategies and resources based on the student's learning style and the course material.
        prompt = (
            "Suggest effective study strategies and resources for the following student learning style and course material:\n\n"
            + student_learning_style + "\n\n"
            + course_material
        )

        # Send the prompt to the LLM and get the response.
        response = await self._aask(prompt)

        # Return the response.
        return response

    async def offer_motivation_and_encouragement_to_stay_engaged_with_the_course(self, student_progress):
        """Offers motivation and encouragement to stay engaged with the course based on the student's progress."""

        # Prompt the LLM to offer motivation and encouragement to stay engaged with the course based on the student's progress.
        prompt = (
            "Offer motivation and encouragement to stay engaged with the course based on the following student progress:\n\n"
            + student_progress
        )

        # Send the prompt to the LLM and get the response.
        response = await self._aask(prompt)

        # Return the response.
        return response

    async def generate_personalized_learning_plans_for_students_based_on_their_individual_needs_and_goals(self, student_needs, student_goals):
        """Generates a personalized learning plan for a student based on their individual"""