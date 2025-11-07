from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Build the prompt template
prompt = PromptTemplate(
    template="Give five intersting facts about {topic}",
    input_variables=["topic"]
)

def get_topic():
    input_topic = input("Enter a topic: ")
    return {"topic": input_topic}

model = ChatOpenAI(model="gpt-5")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke(get_topic())

print(result)

chain.get_graph().print_ascii()