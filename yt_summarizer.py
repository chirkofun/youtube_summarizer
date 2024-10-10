import warnings
warnings.filterwarnings("ignore")

import re
from dotenv import load_dotenv
from langchain_together import ChatTogether
import os
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain import LLMChain
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Initialize the LLM
load_dotenv(override=True)
llm = ChatTogether(api_key=os.getenv("TOGETHER_API_KEY"), temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

def is_youtube_url(url):
    # Use ReGeX to match YouTube URLs
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )

    # Match the URL agains the regext pattern
    match = youtube_regex.match(url)

    return bool(match)

def summarise(video_url):
    # Get a transcript from a video
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    data = loader.load()

    summarizer_template = PromptTemplate(
        input_variables=["video_transcript"],
        template="""
        Read through the entire transcript carefully.
        Provide a concise summary of the video's main topic and purpose.
        Extract and list the five most interesting or important points from the transcript.
        For each point: State the key idea in a clear and concise manner.

        - Ensure your summary and key points capture the essence of the video without including unnecessary details.
        - Use clear, engaging language that is accessible to a general audience.
        - If the transcript includes any statistical data, expert opinions, or unique insights, prioritize including these in your summary or key points.

        VERY IMPORTANT: make sure that your response is within 1600 characters.
        
        Video transcript: {video_transcript}
        """
    )

    # invoke the chain with the transcript
    chain = LLMChain(llm=llm, prompt=summarizer_template)

    summary = chain.invoke({
        "video_transcript": data[0].page_content
    })

    return (summary["text"])

@app.route("/ping", methods=['GET'])
def pinger():
    return "<p>Hello world!</p>"

@app.route("/summary", methods=['POST'])
def summary():
    url = request.form.get('Body') # Get the JSON data from the request body
    print(url)
    if is_youtube_url(url):
        response = summarise(url)
    else:
        response = "Please check if you have sent correct YouTube video URL"

    print(response)
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response)
    return str(resp)

# Run the Flask app
if __name__ == '__main__':
    app.run(port=8888)