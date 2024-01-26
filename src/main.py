import os

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from transformers import pipeline

# Instantiate your chat model with a good template for summarization
llm = HuggingFaceEndpoint(
    endpoint_url=os.environ['LLM_ENDPOINT'],
    task="text2text-generation",
    model_kwargs={
        "max_new_tokens": 200
    }
)
chat_model = ChatHuggingFace(llm=llm)

textInput = """
<|system|>
You are that helpful AI that summarizes a text that contains multiple product reviews.</s>
<|user|>
{userInput}</s>
<|assistant|>
"""


def main():
    print("(GENERIC SENTIMENT ANALYSIS)-------------------------------------------------------------")

    generic_sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    data = ["I love you", "I hate you", "I know you"]
    output = generic_sentiment_pipeline(data)

    # for every piece of data in output, print the sentiment
    for i in range(len(data)):

        print("------------------------------")
        print("INPUT: " + data[i])
        print("------------------------------")

        sentiment = output[i]
        if sentiment['label'] == 'POS':
            print("*The sentiment is POSITIVE with a certainty of: " + str(round(sentiment['score'], 3) * 100) + "%")
        elif sentiment['label'] == 'NEG':
            print("*The sentiment is NEGATIVE with a certainty of: " + str(round(sentiment['score'], 3) * 100) + "%")
        else:
            print("*The sentiment is NEUTRAL with a certainty of: " + str(round(sentiment['score'], 3) * 100) + "%")

    print("(PRODUCT REVIEW SENTIMENT ANALYSIS)---------------------------------------------------------")

    data = ["I bought these balls before and they were like new. This time , "
            "several had signs that they were damaged. Quality control?",
            "For 12 bucks it’s definitely a good deal "
            "but a few of the balls are scuffed and a few have deep scratches",
            "Perfect balls, would recommend. Never had a problem with them.",
            "These balls are horrible! They break when hit.",
            "These are a good deal for used balls. I tend to lose a balm or two per round. "
            "If your not worried about a small blemish and like Kirkland golf balls. Give them a try.",
            "Amazing golf balls, I get these every time I need some. Very rarely are they scuffed"
            "or otherwise damaged",
            ]

    review_sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
    output = review_sentiment_pipeline(data)

    # order the output array by score, highest to lowest
    output = sorted(output, key=lambda x: x['label'], reverse=True)

    # for every piece of data in output, print the sentiment
    for i in range(len(data)):
        print(output[i])
    """This score isn't necessarily for the customer's use. Typically, a customer will put their own star rating.
    However, the stars the user puts is not necessarily an exact sentiment. (ex: "Perfect product" with 4/5 stars)
    This method of review is most useful for INTERNAL use, KPI's etc., as it provides a purely sentiment-based rating."""

    print("(PRODUCT REVIEW SUMMARIZATION)---------------------------------------------------------")

    # convert data to string
    data_string = "\n".join(data)

    # Invoke your chat model defined above
    output = chat_model.invoke(textInput.format(userInput=data_string))

    print(output.content)


if __name__ == '__main__':
    main()
