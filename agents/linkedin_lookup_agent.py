import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from tools.tools import get_profile_url_tavily

def lookup(name: str) -> str:
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0
    )

    template = '''
    Given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
    your answer should contain only a url 
    '''
    prompt_template = PromptTemplate(
        template=template, input_variables=['name_of_person']
        ) 
    
    tools_for_agent = [
        Tool(
            name='Crawl Google 4 linkedin profile page',
            func=get_profile_url_tavily,
            description='useful for when you need to get the Linkedin page URL'
        )
    ]

    react_prompt = hub.pull('hwchase17/react')
    agent = create_react_agent(llm=llm,  tools=tools_for_agent, prompt=react_prompt)
    agentExecutor = AgentExecutor( agent=agent, tools=tools_for_agent, verbose=True)

    result = agentExecutor.invoke(
        input={'input':prompt_template.format_prompt(name_of_person = name)}
    )

    linkedin_profile_url = result['output']
    return linkedin_profile_url

if __name__ == "__main__":
    linkedin_url = lookup(name="Mayank Tiwari")
    print(linkedin_url)