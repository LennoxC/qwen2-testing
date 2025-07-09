from qwen25 import doqwen25
from qwen2 import doqwen2


prompt = '''
This is an image of a wine label. Using the text in the image, answer the following questions: 
What alcohol percentage is the wine?
What varietal is the wine?
What region and country was the wine produced in?
What preservative might be contained?
Answer in JSON format.
'''

doqwen25("https://www.toptastes.co.nz/wp-content/uploads/2018/08/Thornbury-Sauvignon-Blanc-2018-back-label.jpg-642x1030.jpg", prompt)
