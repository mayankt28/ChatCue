from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from third_party.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break_with(name: str) -> str:
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
        )
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0
    )
    chain = summary_prompt_template | llm
    res = chain.invoke({"information":linkedinData})
    return res

if __name__ == "__main__":
    print("Entering Icebreaker")
    ice_break_with(name='Mayank Tiwari')

