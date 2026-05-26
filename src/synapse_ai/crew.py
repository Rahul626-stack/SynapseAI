from dotenv import load_dotenv
from crewai import Agent, LLM, Crew, Task, Process
from crewai.project import agent, task, crew, CrewBase
from synapse_ai.models.quiz import QuizOutput

load_dotenv()


@CrewBase
class SynapseAICrew:
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.7,
    )
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/task.yaml'

    @agent
    def pdf_analyzer(self) -> Agent:
        """Content analyzer agent — receives RAG-retrieved context (no longer needs PDF tool)."""
        return Agent(
            config=self.agents_config['pdf_analyzer'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            # NOTE: ExtractPDFContentTool removed — RAG pipeline handles PDF reading now.
            # The agent receives pre-retrieved context via the task description.
        )

    @agent
    def quiz_generator(self) -> Agent:
        """Quiz generation agent — creates Bloom's-tagged questions from analyzed content."""
        return Agent(
            config=self.agents_config['quiz_generator'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    @task
    def pdf_analyzer_task(self) -> Task:
        return Task(
            config=self.tasks_config['pdf_analyzer_task'],
        )

    @task
    def quiz_generator_task(self) -> Task:
        return Task(
            config=self.tasks_config['quiz_generator_task'],
            output_pydantic=QuizOutput
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.pdf_analyzer(), self.quiz_generator()],
            tasks=[self.pdf_analyzer_task(), self.quiz_generator_task()],
            verbose=True,
            process=Process.sequential,
        )
