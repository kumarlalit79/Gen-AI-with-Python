from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# step 1 (a) -  Indexing (Document Ingestion)
video_id = "9hQBT9nAb5c"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, language=["en"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    
    print(transcript)

except:
    print("No caption available for this video")  