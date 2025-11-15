import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
import os

from typing import List
import operator
from typing import Annotated, Dict, Any

def initialize_env() -> None:
    LANGCHAIN_API_KEY=st.secrets["langchain_key"]
    OPENAI_AI_KEY=st.secrets["open_ai_key"]


    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]="AIClub_Pro"
    os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
    os.environ['OPEN_AI_KEY'] = OPENAI_AI_KEY

def create_llm_msg(system_prompt,history):
    resp=[SystemMessage(content=system_prompt)]
    msgs = history
    for m in msgs:
        if m["role"] == "user":
            resp.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            resp.append(AIMessage(content=m["content"]))
    #print(f"DEBUG CREATE LLM MSGS: {history=}\n{resp=}")
    return resp



class AgentState(BaseModel):
    """State of the agent."""
    messages: list = []
    response: Annotated[List[str], operator.add] = []
    categories: List[str] = []
    combined_response: List[Dict[str, Any]] = []

class Category(BaseModel):
    """Category for the agent."""
    categories: List[str] = []

import yaml

class ChatbotAgent():
    """A chatbot agent that interacts with users."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model_name="gpt-5-nano", openai_api_key=api_key)
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classifier", self.classifier)
        workflow.add_node("aws_agent", self.aws_agent)
        workflow.add_node("azure_agent", self.azure_agent)
        workflow.add_node("gcp_agent", self.gcp_agent)
        workflow.add_node("oci_agent", self.oci_agent)

        # Start to classifier
        workflow.add_edge(START, "classifier")

        # Define a single routing function for the classifier
        # This function will inspect state.categories and return a list of agent nodes to run.
        def route_agents(state: AgentState) -> List[str]:
            next_nodes = []
            if "aws_agent" in state.categories:
                next_nodes.append("aws_agent")
            if "azure_agent" in state.categories:
                next_nodes.append("azure_agent")
            if "gcp_agent" in state.categories:
                next_nodes.append("gcp_agent")
            if "oci_agent" in state.categories:
                next_nodes.append("oci_agent")
            return next_nodes

        # Add conditional edges from classifier using the routing function
        workflow.add_conditional_edges(
            "classifier",
            route_agents
        )

        # End edges from each agent node to END
        for cloud_agent in ["aws_agent", "azure_agent", "gcp_agent", "oci_agent"]:
            workflow.add_edge(cloud_agent, END)

        self.graph = workflow.compile()



    def classifier(self, state: AgentState):
        # There is no need to use an LLM here, since the category is a 1:1 mapping from the input file

        # Parse the YAML input from user messages to get list of substrates
        user_input = None
        for msg in state.messages:
            if msg["role"] == "user":
                user_input = msg["content"]
                break

        substrates = []
        try:
            parsed_yaml = yaml.safe_load(user_input)
            # Assuming YAML structure has a 'substrates' or 'clouds' list
            substrates = parsed_yaml.get("substrates", []) or parsed_yaml.get("clouds", [])
        except Exception as e:
            print(f"YAML parsing error: {e}")

        # Map substrate names to category agent names
        category_map = {
            "aws": "aws_agent",
            "azure": "azure_agent",
            "gcp": "gcp_agent",
            "oci": "oci_agent",
        }
        categories = [category_map.get(s.lower(), "aws_agent") for s in substrates]

        print(f"Classified categories: {categories}")
        # Return list of categories instead of single one
        return {"categories": categories}


    def main_router(self, state: AgentState):
        # Expecting a list of categories (sub-agents) to run
        categories = getattr(state, "categories", [])
        responses = []

        for category in categories:
            if category == "aws_agent":
                resp = self.aws_agent(state)
                print(f"AWS sub-agent response is {resp}")
            elif category == "azure_agent":
                resp = self.azure_agent(state)
                print(f"Azure sub-agent response is {resp}")
            elif category == "gcp_agent":
                resp = self.gcp_agent(state)
                print(f"GCP sub-agent response is {resp}")
            elif category == "oci_agent":
                resp = self.oci_agent(state)
                print(f"OCI sub-agent response is {resp}")
            else:
                resp = self.aws_agent(state)  # default fallback
                print(f"Default sub-agent response is {resp}")

            responses.append(resp)

        state.combined_response = responses
        print(f"Main router responses: {state.combined_response}")
        return state.categories
        # Combine or aggregate multiple responses
        # For example, concatenate stream responses or manage aggregation as needed
        # return {"combined_response": responses}

    def aws_agent(self, state: AgentState):
        print("AWS agent processing....")
        AWS_PROMPT = f"""
        You are an AWS agent that generates AWS SQS code to create a standard or FIFO queue, add a message to a queue,
        delete a message from a queue, get a message from the queue, or change its visibility timeout or number of retries.
        Given a user provided YAML file that describes the queue, please generate the AWS terraform code to
        provision the queue and Python code using the Boto SDK to interact with the queue.
        """
        llm_messages = create_llm_msg(AWS_PROMPT, state.messages)
        # Collect streamed output into a list of strings
        collected_response = []
        for chunk in self.model.stream(llm_messages):
            content = getattr(chunk, "content", "")
            if content:
                collected_response.append(content)
        return {"response": collected_response, "category": "aws_agent"}

    def azure_agent(self, state: AgentState):
        print("Azure agent processing....")
        AZURE_PROMPT = f"""
        You are an Azure agent that generates Azure code to create a Azure storage queue or Azure service bus queue, add a message to a queue,
        delete a message from a queue, get a message from the queue, or change its visibility timeout or number of retries.
        Given a user provided YAML file that describes the queue, please generate the Azure terraform code to
        provision the queue and Python code using the azure-storage-queue or azure-servicebus SDK to interact with the queue.
        """
        llm_messages = create_llm_msg(AZURE_PROMPT, state.messages)
        # Collect streamed output into a list of strings
        collected_response = []
        for chunk in self.model.stream(llm_messages):
            content = getattr(chunk, "content", "")
            if content:
                collected_response.append(content)
        return {"response": collected_response, "category": "azure_agent"}

    def gcp_agent(self, state: AgentState):
        print("GCP agent processing....")
        GCP_PROMPT = f"""
        You are a GCP agent that generates GCP code to create a GCP cloud pub/sub, add a message to a queue,
        delete a message from a queue, get a message from the queue, or change its visibility timeout or number of retries.
        Given a user provided YAML file that describes the queue, please generate the GCP terraform code to
        provision the queue and Python code using the google-cloud-pubsub SDK to interact with the queue.
        """
        llm_messages = create_llm_msg(GCP_PROMPT, state.messages)
        # Collect streamed output into a list of strings
        collected_response = []
        for chunk in self.model.stream(llm_messages):
            content = getattr(chunk, "content", "")
            if content:
                collected_response.append(content)
        return {"response": collected_response, "category": "gcp_agent"}

    def oci_agent(self, state: AgentState):
        print("OCI agent processing....")
        OCI_PROMPT = f"""
        You are an OCI agent that generates OCI code to create a OCI queue service, add a message to a queue,
        delete a message from a queue, get a message from the queue, or change its visibility timeout or number of retries.
        Given a user provided YAML file that describes the queue, please generate the OCI terraform code to
        provision the queue and Python code using the oci.queue.QueueClient SDK to interact with the queue.
        """
        llm_messages = create_llm_msg(OCI_PROMPT, state.messages)
        # Collect streamed output into a list of strings
        collected_response = []
        for chunk in self.model.stream(llm_messages):
            content = getattr(chunk, "content", "")
            if content:
                collected_response.append(content)
        return {"response": collected_response, "category": "oci_agent"}

def generate_substrate_code() -> None:
    # Single message
    msg_list=[]

    # Ask user to enter file path
    file_path = input("Enter the YAML file path describing the queue: ")

    if not os.path.exists(file_path):
        raise FileNotFoundError("The specified file does not exist.")

    # Read YAML file content as a string
    with open(file_path, 'r', encoding='utf-8') as f:
        user_input = f.read()

    print(f"User input is {user_input}")
    # create YAML string to describe the input

    msg_list.append({"role":"user","content":user_input})
    app=ChatbotAgent(os.environ['OPEN_AI_KEY'])
    thread_id = 10
    thread={"configurable":{"thread_id":thread_id}}
    full_resp = ""
    for s in app.graph.stream({'messages': msg_list}, thread):
        print(f"Inside the for loop in the app graph stream s contains {s}")
        for key, value in s.items():
            print(f"Key: {key}, Value: {value}")
            cloud_type = key
            response_generator = value.get("response")
            if response_generator:
                #print(f"response_generator is {response_generator}")
                full_resp_for_one_cloud = "".join(response_generator)
                print(f"Obtained full response for {cloud_type}")
                output_path = input(f"Enter the absolute output path that will contain {cloud_type}.out substrate: ")
                #print(f"full response for {cloud_type} is {full_resp_for_one_cloud}")
                # open an output file called <cloud_type>.out
                # and dump the contents to that file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(f"{output_path}/{cloud_type}.out", "w", encoding="utf-8") as file:
                    file.write(full_resp_for_one_cloud)
      
st.title("Welcome to the Cloud Abstrator")
st.write(
    "We help you to abstract the cloud"
)
initialize_env()
generate_substrate_code()

