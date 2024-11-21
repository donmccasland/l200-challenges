from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from . import TravelChatBot



class GoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Vertex AI Model"

chatbot = TravelChatBot.TravelChatBot()

# initiatialize the  wrapper class
vertexai_gemini = GoogleVertexAI(model=chatbot.model())

def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(model=vertexai_gemini, threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    assert_test(test_case, [answer_relevancy_metric])
