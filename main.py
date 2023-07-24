from typing import Sequence
from google.cloud import vision
import vertexai

import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.preview.language_models import (ChatModel, InputOutputTextPair,TextEmbeddingModel,TextGenerationModel)




def analyze_image_from_uri(
    image_uri: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = image_uri
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response


def print_labels(response: vision.AnnotateImageResponse):
    print("=" * 80)
    for label in response.label_annotations:
        print(
            f"{label.score:4.0%}",
            f"{label.description:5}",
            sep=" | ",
        )

def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    extracted_text=[]

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        
        extracted_text += {text.description}

    return extracted_text
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

employment_letter = "gs://employment-letter1/Employee-Confirmation-Letter.png"
text_detected = detect_text_uri(employment_letter)
employment_letter_content = text_detected[0]

PROJECT_ID = "cascades-visionapi"
LOCATION = "us-central1" #e.g. us-central1
vertexai.init(project=PROJECT_ID, location=LOCATION)
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

text = f"Full name, Current address, Date submitted, Employer,Employer address, Hiring date,Job title,Employment status, if any of the information is missing, say that is missed, {employment_letter_content}"
prompt = "extract these data points from the text given"
response = generation_model.predict(prompt=f"{prompt}:{text}")

print(response.text)

employment_letter2 = "gs://employment-letter1/Confirmation-of-Employment-Letter-Example.jpg"

def detect_logos_uri(uri):
    """Detects logos in the file located in Google Cloud Storage or on the Web."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    print("Logos:")

    for logo in logos:
        print(logo.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )



def extract_info(event, context):
  #data = cloud_event.data
  bucket = event["bucket"]
  name = event["name"]
  uri = "gs://"+ bucket +"/"+ name 
  text_detected = detect_text_uri(uri)
  employment_letter_content = text_detected[0]
  generation_model = TextGenerationModel.from_pretrained("text-bison@001")
  text = f"Full name, Current address, Date submitted, Employer,Employer address, Hiring date,Job title,Employment status, if any of the information is missing, say that is missed, {employment_letter_content}"
  prompt = "extract these data points from the text given"
  response = generation_model.predict(prompt=f"{prompt}:{text}")
  print(response.text)
  detect_logos_uri(uri)
