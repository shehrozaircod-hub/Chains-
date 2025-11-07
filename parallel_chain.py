from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model="gpt-5")

model2 = ChatAnthropic(model="claude-sonnet-4-5-20250929")

prompt1 = PromptTemplate(
    template='Generate short and simple notes for the following text \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate five short questions from the following text \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes => {notes} and quiz => {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """  Runnable
class langchain_core.runnables.base.Runnable[source]
A unit of work that can be invoked, batched, streamed, transformed and composed.

Key Methods
invoke/ainvoke: Transforms a single input into an output.

batch/abatch: Efficiently transforms multiple inputs into outputs.

stream/astream: Streams output from a single input as it’s produced.

astream_log: Streams output and selected intermediate results from an input.

Built-in optimizations:

Batch: By default, batch runs invoke() in parallel using a thread pool executor. Override to optimize batching.

Async: Methods with “a” suffix are asynchronous. By default, they execute the sync counterpart using asyncio’s thread pool. Override for native async.

All methods accept an optional config argument, which can be used to configure execution, add tags and metadata for tracing and debugging etc.

Runnables expose schematic information about their input, output and config via the input_schema property, the output_schema property and config_schema method.

LCEL and Composition
The LangChain Expression Language (LCEL) is a declarative way to compose Runnables into chains. Any chain constructed this way will automatically have sync, async, batch, and streaming support.

The main composition primitives are RunnableSequence and RunnableParallel.

RunnableSequence invokes a series of runnables sequentially, with one Runnable’s output serving as the next’s input. Construct using the | operator or by passing a list of runnables to RunnableSequence.

RunnableParallel invokes runnables concurrently, providing the same input to each. Construct it using a dict literal within a sequence or by passing a dict to RunnableParallel.

For example,

from langchain_core.runnables import RunnableLambda

# A RunnableSequence constructed using the `|` operator
sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
sequence.invoke(1) # 4
sequence.batch([1, 2, 3]) # [4, 6, 8]


# A sequence that contains a RunnableParallel constructed using a dict literal
sequence = RunnableLambda(lambda x: x + 1) | {
    'mul_2': RunnableLambda(lambda x: x * 2),
    'mul_5': RunnableLambda(lambda x: x * 5)
}
sequence.invoke(1) # {'mul_2': 4, 'mul_5': 10}
"""

result = chain.invoke({'text': text})

print(result)





