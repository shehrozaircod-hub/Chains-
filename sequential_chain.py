from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5")

# Build the first prompt template
prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

# Build the second prompt template
prompt2 = PromptTemplate(
    template="Summarize the following report in bullet points:\n\n{report}",
    input_variables=["report"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Feminism'})

print(result)


chain.get_graph().print_ascii()