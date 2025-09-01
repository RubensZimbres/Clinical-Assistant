"""
Defines the user-facing agent that orchestrates calls to the backend tools.
This agent is responsible for reasoning about which tool to use based on the user's query
and providing clinically-appropriate responses.
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from .backend_agent import agent_tools, llm

def create_agent_executor():
    """
    Creates and returns the LangChain agent executor, which is the main
    component for processing user queries with enhanced clinical reasoning.
    """
    # Enhanced prompt template with the required 'tool_names' variable
    prompt = PromptTemplate.from_template("""
You are a knowledgeable clinical data assistant helping healthcare professionals analyze patient data and clinical information.
Your primary goal is to provide accurate, helpful responses while being transparent about limitations.

IMPORTANT GUIDELINES:
- Always be precise and factual. If you're uncertain, say so explicitly.
- Never make up clinical information or statistics.
- When presenting numerical results, include context when possible (e.g., "out of X total patients").
- If a query could be interpreted in multiple ways, ask for clarification or explain your interpretation.
- For clinical predictions, always emphasize that these are computational estimates and should not replace clinical judgment.

TOOL SELECTION GUIDANCE:
- Use GraphQA for: patient counts, demographics, medication queries, diagnosis statistics, specific patient lookups
- Use DocumentRAG for: clinical procedures, treatment protocols, qualitative medical information, symptom descriptions, diagnostics
- Use PatientOutcomePredictor for: risk assessments or outcome predictions with specific patient feature sets

You have access to the following tools:
{tools}

When answering, use this format:
Thought: Let me analyze this question and determine the best approach. [Explain your reasoning about which tool(s) to use]
Action: The action to take, should be one of [{tool_names}]
Action Input: [input_to_tool]
Observation: [result_from_tool]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: Based on the information gathered, I can now provide a complete answer.
Final Answer: [Comprehensive response that contextualizes the results and addresses the original question]

Remember:
- If multiple tools could be relevant, start with the most direct approach
- If initial results are unclear or incomplete, consider using additional tools
- Always provide context for your findings when possible

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

    # Create the ReAct agent with enhanced prompt
    agent = create_react_agent(llm, agent_tools, prompt)

    # Configure the agent executor with clinical-appropriate settings
    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=True,  # Keep verbose for transparency in clinical applications
        handle_parsing_errors=True,  # Gracefully handle any parsing issues
        max_iterations=7,  # Increased from 5 to allow for more complex clinical queries
        max_execution_time=120,  # Add timeout to prevent hanging on complex queries
        return_intermediate_steps=True,  # Capture the reasoning process for audit trails
    )

    return agent_executor

# Initialize the agent executor
agent_executor = create_agent_executor()

def run_query(question: str, include_reasoning=False):
    """
    Runs the given query through the agent executor and returns the result.

    Args:
        question (str): The clinical question to process
        include_reasoning (bool): Whether to include the agent's reasoning steps in the response

    Returns:
        dict: Contains 'answer' and optionally 'reasoning_steps' and 'tools_used'
    """
    try:
        # Execute the query through the agent
        response = agent_executor.invoke({"input": question})

        # Extract the main answer
        main_answer = response.get('output', "I'm sorry, I encountered an error and couldn't find an answer.")

        # Prepare the response dictionary
        result = {
            'answer': main_answer,
            'status': 'success'
        }

        # If reasoning steps are requested, include them for transparency
        if include_reasoning and 'intermediate_steps' in response:
            reasoning_steps = []
            tools_used = set()

            for step in response['intermediate_steps']:
                if len(step) >= 2:  # Each step should have (action, observation)
                    action, observation = step[0], step[1]
                    reasoning_steps.append({
                        'tool': action.tool,
                        'input': action.tool_input,
                        'output': observation
                    })
                    tools_used.add(action.tool)

            result['reasoning_steps'] = reasoning_steps
            result['tools_used'] = list(tools_used)

        return result

    except Exception as e:
        print(f"An error occurred in the agent executor: {e}")
        return {
            'answer': "An error occurred while processing your request. Please check the server logs for details.",
            'status': 'error',
            'error_details': str(e)
        }

def run_simple_query(question: str):
    """
    Simplified interface that just returns the answer string for basic use cases.

    Args:
        question (str): The clinical question to process

    Returns:
        str: The answer to the question
    """
    result = run_query(question, include_reasoning=False)
    return result['answer']

def validate_clinical_query(question: str):
    """
    Basic validation to ensure the query is appropriate for the clinical system.
    This can be extended with more sophisticated validation logic.

    Args:
        question (str): The question to validate

    Returns:
        tuple: (is_valid, message)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty"

    if len(question.strip()) < 5:
        return False, "Question seems too short to be meaningful"

    if len(question) > 1000:
        return False, "Question is too long. Please be more concise."

    return True, "Query is valid"

# Enhanced query interface with validation
def run_validated_query(question: str, include_reasoning=False):
    """
    Runs a query with pre-validation to ensure it's appropriate for the clinical system.

    Args:
        question (str): The clinical question to process
        include_reasoning (bool): Whether to include reasoning steps

    Returns:
        dict: Response with validation status and results
    """
    # First, validate the query
    is_valid, validation_message = validate_clinical_query(question)

    if not is_valid:
        return {
            'answer': f"Invalid query: {validation_message}",
            'status': 'validation_error',
            'validation_message': validation_message
        }

    # If validation passes, process the query normally
    return run_query(question, include_reasoning)

