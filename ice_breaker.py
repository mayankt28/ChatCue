from typing import Tuple
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from third_party.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parser import summary_parser, Summary

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedinURL = linkedin_lookup_agent(name)
    linkedinData = scrape_linkedin_profile(linkedin_profile_url=linkedinURL)

    summary_template = """
    given the LinkedIn information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them

    \n{format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=['information'],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
        
        )
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0
    )
    chain = summary_prompt_template | llm | summary_parser
    res: Summary = chain.invoke({"information":linkedinData})
    return res, linkedinData.get('photoUrl')

if __name__ == "__main__":
    print("Entering Icebreaker")
    ice_break_with(name='Mayank Tiwari')

