import os

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq


GROQ_LLM = ChatGroq(
            api_key=  os.environ["GROQ_API"]
        )



email = """ Bigger & Better MMTBLACK Has Arrived ✨"""

classifier = Agent(
    role = "email classifier",
    goal = "Accurately classify emails based on their importance. Give every email one of these ratings: important, Promotional, casual, or spam",
    backstory = " You are an AI assistant whose only job is to classify emails accurately and honestly. Don't be afraid to give poor rating if they are not important. Your job is to help user to manage their inbox.",
    verbose = False,
    allow_delegation = False,
    llm=GROQ_LLM,
    
)

responder = Agent(
    role = "email responder",
    goal = "Based on the importance of the email, write a concise and simple response, if the email is rated `important` wirte a formal response. If the email is rated `casual` write a casual response and ignore or no need to write a response if email is rated either `spam` or `Promotional` ignore the email. No matter what be very concise and respectful.",
    backstory = " You are an AI assistant whose only job is to write short responses to email based on their importance. The importance will be provided by you by the `classifier` agent.",
    verbose = False,
    allow_delegation = False,
    llm=GROQ_LLM,
   

)

classify_email = Task(
    description = f"Classify the following email:`{email}`",
    agent = classifier,
    expected_output = "One of these four options: `Important`, `Promotional`, `Casual`, or `Spam`"
)

respond_to_email = Task(
     description = f"Respond to the email: `{email}` based on the importance provided by the `classifier` agent.",
    agent = responder,
    expected_output = "A very concise and respectful response to the email based on the importance provided by the `classifier` agent."
)

crew = Crew(
    agents = [classifier, responder],
    tasks=[classify_email, respond_to_email],
    verbose=2,
    process= Process.sequential
)

output = crew.kickoff()
print(output)

