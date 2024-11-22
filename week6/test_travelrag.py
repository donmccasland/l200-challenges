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
        ai_msg = chat_model.invoke({"input": prompt, "chat_history":[]})
        return ai_msg["answer"]

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        ai_msg = await chat_model.ainvoke({"input": prompt, "chat_history":[]})
        return ai_msg["answer"]

    def get_model_name(self):
        return "Langchain Vertex AI Model"

chatbot = TravelChatBot.TravelChatBot()

# initiatialize the  wrapper class
vertexai_gemini = GoogleVertexAI(model=chatbot.model())

def test_florence_hotel_vectors():
    answer_relevancy_metric = AnswerRelevancyMetric(model=vertexai_gemini, threshold=0.5, verbose_mode=True)
    actual_output = vertexai_gemini.generate("Can you help me find a hotel in Florence with a great wine experience?")
    test_case = LLMTestCase(
        input="Can you help me find a hotel in Florence with a great wine experience?",
        actual_output=actual_output,
        retrieval_context=["The Florence Wine Hotel"]
    )
    assert_test(test_case, [answer_relevancy_metric])


def test_travel_intro():
    answer_relevancy_metric = AnswerRelevancyMetric(model=vertexai_gemini, threshold=0.5, verbose_mode=True)
    actual_output = vertexai_gemini.generate("Can you help me plan a trip to Paris?")
    test_case = LLMTestCase(
        input="Can you help me plan a trip to Paris?",
        actual_output=actual_output,
        retrieval_context=["Yes.  When do you want to go?  What is your budget?  How many people are going?"]
    )
    assert_test(test_case, [answer_relevancy_metric])
#Human: Hi.  My family and I are taking a trip to Paris.  Can you help us plan it?
#Bot: I can definitely help you plan your trip to Paris!  To give you the best recommendations, I need a bit more information about your family and what you'd like to do.  
#
#Can you tell me:
#
#* **How many people are in your family?**  (Adults and children, and their ages?)
#* **What are your interests?**  (Art museums, history, food, shopping, nightlife, etc.)
#* **What is your budget?** (This will help me recommend appropriate accommodation and activities)
#* **What time of year are you planning to travel?** (This will affect weather and crowds)
#* **How long are you planning to stay?** (This will determine how much we can fit in)
#
#Once I have this information, I can create a personalized itinerary for your family's trip to Paris! 
#
#Human: There are 3 people in our family.  We are interested in seeing museums and eating great pastries.  We are hoping to go in late spring around May and want to stay for two weeks.
#Bot: That sounds like a fantastic trip!  Two weeks in Paris in late spring will give you ample time to explore its museums and indulge in delicious pastries.  Here’s a preliminary plan to get you started:
#
#**Accommodation:**
#
#* **Location:**  Consider staying in the Latin Quarter or Marais district.  These neighborhoods are centrally located, close to many museums, charming cafes, and pastry shops. 
#* **Type:**  For a family, a comfortable apartment with a kitchen could be ideal.  You can enjoy some meals in and save money.
#
#**Museums:**
#
#* **Must-Sees:**  The Louvre (Mona Lisa!), Musée d'Orsay (Impressionist masterpieces), Centre Pompidou (modern art), Musée Rodin (sculptures), and the Musée Picasso are all excellent choices. 
#* **Tips:**  Buy museum passes online in advance to save time and avoid lines.
#* **Alternative:**  The Musée Carnavalet (history of Paris) or the Musée du Quai Branly - Jacques Chirac (non-European art) are great for a different perspective. 
#
#**Pastries:**
#
#* **Famous Patisseries:**  Ladurée (macarons!), Pierre Hermé (modern pastries), and Poilâne (bread) are highly regarded.
#* **Hidden Gems:**  Ask locals for recommendations for smaller bakeries in your neighborhood, often serving delicious, traditional pastries.
#* **Try a croissant or pain au chocolat:**  Don’t forget to try a traditional Parisian breakfast!
#
#**Other Activities:**
#
#* **Seine River Cruise:**  A scenic way to see the city.
#* **Eiffel Tower:**  Take an elevator to the top for panoramic views.
#* **Montmartre:**  Visit the Sacré-Coeur basilica and explore the charming streets.
#* **Père Lachaise Cemetery:**  A fascinating place to visit the graves of famous artists and writers.
#
#**Day Trips:**
#
#* **Versailles Palace:**  Just a short train ride from Paris, this is a must-see for history buffs.
#* **Giverny:**  Home to Claude Monet's gardens, a beautiful place to visit in spring.
#
#**Budget:**
#
#* **Accommodation:**  Apartments can be more affordable than hotels.
#* **Food:**  Pastries can be expensive, but you can find budget-friendly options at markets and bakeries.
#* **Transportation:**  The Metro is a cost-effective way to get around.
#
#**Things to consider:**
#
#* **Reservations:**  Book museum tickets, restaurant tables, and day trip transportation in advance, especially during peak season.
#* **Weather:**  May weather in Paris can be unpredictable.  Be prepared for cool mornings and evenings, and pack layers.
#* **Language:**  While many Parisians speak English, learning a few basic French phrases will enhance your experience.
#
#**Next Steps:**
#
#* **Further research:**  Use travel websites and blogs to get more detailed information on each activity.
#* **Create a rough itinerary:**  Plan your days, factor in travel time, and decide on your must-see attractions.
#* **Book accommodation and transportation:**  Secure your lodging and train tickets to avoid disappointment.
#
#I hope this gets you started!  Let me know if you have any further questions. 
