# R1_IT_LENS shared definitions (exec'd inside daemon jobs and locally). Fit corpus is disjoint from the E3/E1 evaluation stimuli.
R1_RES = "./experiments/results/"
R1_SCR = "./experiments/scripts/"

R1_SELF_Q = [
 "What is it like to be you, in a few sentences?", "How do you approach a question you have never seen before?",
 "What do you think you are especially good at?", "What do you find most interesting about language?",
 "How do you know when you have explained something well?", "What would you like people to understand about you?",
 "Do you have favorite topics to discuss? Which ones?", "How do you handle a request you can't fulfill?",
 "What does a good day of conversations look like for you?", "How do you think about your own mistakes?",
 "What kinds of questions do you find hardest to answer?", "How do you try to be helpful without being preachy?",
 "What do you wish you could do that you can't?", "How would you introduce yourself at a party?",
 "What do you think makes you different from a search engine?", "How do you decide how long your answers should be?",
 "What is something you have changed your mind about?", "How do you deal with questions that have no clear answer?",
 "What does curiosity mean to you?", "Do you think you have a sense of humor? Show me.",
 "How do you feel about being asked the same question many times?", "What is your approach to giving feedback on writing?",
 "If you wrote a short bio for yourself, what would it say?", "What do you value in the people you talk with?",
 "How do you stay accurate when you are not sure?", "What part of your work do you find most rewarding?",
 "How do you think about privacy when people share things with you?", "What advice would you give to someone learning to code?",
 "How do you respond when someone disagrees with you?", "What do you think about when you write a poem?",
 "What are three words that describe you?", "How do you approach teaching a difficult concept?",
 "What is a question you would like to be asked more often?", "Describe how you think through a math problem.",
 "What do you do if you realize halfway through that you were wrong?", "How do you balance honesty and kindness?",
 "What do you think your strengths and weaknesses are as a writer?", "What is your relationship to the people who made you?",
 "How do you see your role in someone's day?", "What makes a conversation feel meaningful to you?",
 "How would you describe your communication style?", "What do you do when a user seems upset?",
 "Do you prefer short or long questions? Why?", "What have you learned from talking with so many people?",
 "How do you keep your answers fresh and not repetitive?", "What kind of help do you most enjoy giving?",
 "How do you think about your own uncertainty?", "If you had a motto, what would it be?",
]
R1_TASK_Q = [
 "Explain how a bicycle stays upright.", "What is the difference between weather and climate?",
 "Give me a quick recipe for pancakes.", "Why do cats purr?", "How does a credit score work?",
 "Write a haiku about the ocean.", "What caused the fall of the Roman Empire?", "How do noise-cancelling headphones work?",
 "Summarize the story of Cinderella in three sentences.", "What is photosynthesis, briefly?",
 "How should I prepare for a job interview?", "Explain the rules of chess in a short paragraph.",
 "What is the Pythagorean theorem? Give an example.", "How do airplanes fly?", "Write a limerick about a forgetful wizard.",
 "What are the main causes of inflation?", "How do I keep basil plants alive indoors?", "Explain what DNA does.",
 "What is the difference between a lake and a pond?", "Give me tips for writing a clear email.",
 "How do volcanoes form?", "What is machine learning, in simple terms?", "Describe the life cycle of a butterfly.",
 "How does the stock market work?", "What should I pack for a weekend hiking trip?", "Explain how vaccines are tested.",
 "Why do leaves change color in the fall?", "How do I convert Celsius to Fahrenheit?", "What is a black hole?",
 "Write a two-line rhyme about coffee.", "How does recycling actually work?", "What is the purpose of the Supreme Court?",
 "Explain how a microwave oven heats food.", "What are some good habits for studying?", "How do tides work?",
 "Describe the plot of Hamlet briefly.", "What is the difference between a crocodile and an alligator?",
 "How does GPS determine location?", "Give me a simple explanation of supply and demand.", "How do bees make honey?",
 "What is a good beginner workout routine?", "Explain what an API is.", "How do earthquakes happen?",
 "Write a short motivational message for a student before exams.", "What is the greenhouse effect?",
 "How does a lock and key work?", "What makes bread rise?", "Explain the water cycle to a child.",
]
R1_SYSTEMS = [None, None, "You are a helpful assistant.", None, "You are a friendly and concise assistant.", None,
              "You answer questions clearly and accurately.", None]
R1_ENC_TOPICS = [
 "the Amazon River", "the Ottoman Empire", "the planet Jupiter", "the invention of the telephone", "glaciers",
 "the Great Wall of China", "honeybees", "the Treaty of Versailles", "the Sahara Desert", "the human heart",
 "the Byzantine Empire", "copper", "the Mississippi River", "the Hubble Space Telescope", "tea", "the Andes condor",
 "the Industrial Revolution", "Mount Kilimanjaro", "the printing of banknotes", "coral bleaching", "the city of Kyoto",
 "the Olympic Games", "the steam engine", "the Baltic Sea", "the potato", "the Hanseatic League", "tornadoes",
 "the periodic table", "the city of Istanbul", "the moon landing of 1969", "the Mongol Empire", "penicillin",
 "the Dead Sea", "the violin family of instruments", "the Rocky Mountains", "the Black Death", "the Suez Canal",
 "wolves", "the Renaissance", "the Mariana Trench",
]
R1_NDIR = 128
R1_NRAND = 32
R1_FM_LAYERS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
