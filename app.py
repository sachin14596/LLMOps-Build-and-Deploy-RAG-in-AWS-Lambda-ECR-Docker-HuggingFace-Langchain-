import json
import os
import sys
import boto3
import streamlit as streamlit

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain.vectorstores import FAISS
