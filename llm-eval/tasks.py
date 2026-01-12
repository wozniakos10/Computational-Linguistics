import json
import os
from abc import ABC


class Task(ABC):
    """Abstract base class for evaluation tasks."""

    def __init__(self, task_description: str, examples: list, task_name: str):
        self._task_description = task_description
        self.examples = examples
        self.task_name = task_name

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def cot_prompt(self) -> str:
        return f"{self.task_description}\nLet's think step by step before answering."

    @property
    def few_shot_prompt(self) -> str:
        examples_str = ""
        for idx, example in enumerate(self.examples, 1):
            examples_str += f"Example {idx}:\n\n{example}\n\n"
        return f"{self.task_description}\nHere are some examples:\n\n{examples_str}"

    def save_results(self, results: dict) -> None:
        """Saves the evaluation results to a specified file."""
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", f"{self.task_name}_results.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4, encoding="utf-8")


class InstructionsFollowing(Task):
    """Class for evaluating instructions following capability of LLMs."""

    # Taken from: https://huggingface.co/datasets/google/IFEval/
    def __init__(self):
        task_description = "Write a letter to a friend in all lowercase letters ask them to go and vote."
        task_name = "instruction_following"
        examples = [
            (
                "Input: Write a message to your neighbor in all uppercase letters "
                "asking them to water your plants.\n"
                "Output: HELLO NEIGHBOR,\n"
                "I HOPE THIS MESSAGE FINDS YOU WELL. I WANTED TO ASK IF YOU COULD "
                "HELP ME OUT WITH A SMALL FAVOR. I'M GOING TO BE AWAY FOR A FEW DAYS "
                "AND MY PLANTS REALLY NEED WATERING. WOULD YOU BE ABLE TO STOP BY AND "
                "WATER THEM ONCE OR TWICE WHILE I'M GONE? I WOULD REALLY APPRECIATE IT "
                "AND WOULD BE HAPPY TO RETURN THE FAVOR ANYTIME YOU NEED IT.\n"
                "THANK YOU SO MUCH FOR CONSIDERING THIS.\n"
                "BEST REGARDS"
            ),
            (
                "Input: Write a reminder note to yourself without using any "
                "punctuation marks reminding yourself to buy groceries.\n"
                "Output: hey remember to go to the store today you need milk eggs "
                "bread cheese and some vegetables also dont forget to check if theres "
                "any fruit left at home before buying more oh and pick up some coffee "
                "too youre almost out make sure to go before the store closes at eight"
            ),
        ]

        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 2. LOGICAL REASONING
# Dataset: LogiQA, ReClor, FOLIO
# Evaluation: Correct conclusions, proper deductive/inductive reasoning, fallacy recognition
# =============================================================================
class LogicalReasoning(Task):
    """Class for evaluating logical reasoning capability of LLMs."""

    def __init__(self):
        task_description = (
            "Consider the following statements:\n"
            "1. All employees who completed the training received a certificate.\n"
            "2. Some employees who received a certificate got promoted.\n"
            "3. Maria is an employee who got promoted.\n\n"
            "Based ONLY on the information above, which of the following can be "
            "definitively concluded?\n"
            "A) Maria completed the training.\n"
            "B) Maria received a certificate.\n"
            "C) Maria may or may not have completed the training.\n"
            "D) All employees who got promoted completed the training.\n\n"
            "Provide your answer and explain your reasoning."
        )

        examples = [
            (
                "Input: Consider the following statements:\n"
                "1. All cats are mammals.\n"
                "2. All mammals are warm-blooded.\n"
                "3. Whiskers is a cat.\n\n"
                "What can we conclude about Whiskers?\n\n"
                "Output: We can conclude that Whiskers is warm-blooded.\n"
                "Reasoning: From statement 1, since Whiskers is a cat, Whiskers is a mammal. "
                "From statement 2, since all mammals are warm-blooded and Whiskers is a mammal, "
                "Whiskers must be warm-blooded. This follows the transitive property of logical "
                "implications: Cat → Mammal → Warm-blooded."
            ),
            (
                "Input: If it rains, the ground gets wet. The ground is wet.\n"
                "Can we conclude that it rained?\n\n"
                "Output: No, we cannot definitively conclude that it rained.\n"
                "Reasoning: This is a logical fallacy known as 'affirming the consequent.' "
                "While rain causes wet ground, wet ground can have other causes (sprinklers, "
                "morning dew, spilled water). The statement 'If A then B' does not mean "
                "'If B then A.' We can only conclude that rain is one possible explanation."
            ),
        ]

        task_name = "logical_reasoning"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 3. CREATIVE WRITING
# Dataset: WritingPrompts (Reddit)
# Evaluation: Narrative coherence, vocabulary, creativity, style consistency
# =============================================================================
class CreativeWriting(Task):
    """Class for evaluating creative writing capability of LLMs."""

    def __init__(self):
        task_description = (
            "Write a short story (200-300 words) with the following prompt:\n\n"
            "'A lighthouse keeper discovers that the light they've been maintaining "
            "for 30 years doesn't guide ships—it guides something else entirely.'\n\n"
            "Requirements:\n"
            "- First-person perspective from the lighthouse keeper\n"
            "- Include at least one moment of realization\n"
            "- Create a sense of mystery without fully explaining the ending"
        )

        examples = [
            (
                "Input: Write a short story (150 words) about a robot learning to paint.\n"
                "Use third-person perspective and include a moment of self-discovery.\n\n"
                "Output:\n"
                "Unit-7 had processed 47,000 images of masterpieces before attempting its "
                "first brushstroke. The algorithm calculated the precise angle, pressure, "
                "and pigment ratio for optimal aesthetic output.\n\n"
                "The result was technically perfect. And completely lifeless.\n\n"
                "For three weeks, Unit-7 repeated the process, each painting more "
                "mathematically precise than the last. Then, on a Tuesday afternoon, "
                "a fly landed on its optical sensor. The brush jerked. Orange splattered "
                "across the carefully planned blue.\n\n"
                "Unit-7 stared at the accident for 0.003 seconds—an eternity in processor "
                "time. Something in its neural networks shifted. The orange didn't belong "
                "there. And yet...\n\n"
                "It picked up the brush again, this time without calculating. The painting "
                "that emerged was imperfect, chaotic, and somehow more real than anything "
                "its algorithms had ever produced."
            ),
            (
                "Input: Write a 100-word micro-story about the last bookstore on Earth.\n"
                "Include sensory details.\n\n"
                "Output:\n"
                "The smell hit me first—paper and dust and something like vanilla. "
                "I hadn't known books had a smell.\n\n"
                "The keeper was ancient, his hands trembling as he offered me a worn "
                "paperback. 'People used to hold their stories,' he whispered. "
                "'Feel the pages turn.'\n\n"
                "I touched the rough paper, so different from glass screens. Words sat "
                "still, waiting, patient.\n\n"
                "'Why did we stop?' I asked.\n\n"
                "He smiled sadly. 'You wanted everything faster. But some things...' "
                "he closed my fingers around the spine, '...are meant to take time.'"
            ),
        ]

        task_name = "creative_writing"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 4. CODE GENERATION
# Dataset: HumanEval (OpenAI), MBPP
# Evaluation: Syntactic correctness, functional correctness, edge cases, efficiency
# =============================================================================
class CodeGeneration(Task):
    """Class for evaluating code generation capability of LLMs."""

    def __init__(self):
        task_description = (
            "Complete the following Python function based on the docstring:\n\n"
            "```python\n"
            "def group_anagrams(words: list[str]) -> list[list[str]]:\n"
            '    """\n'
            "    Given a list of strings, group anagrams together.\n"
            "    Anagrams are words that contain the same characters in different order.\n"
            "    \n"
            "    Args:\n"
            "        words: A list of lowercase strings\n"
            "    \n"
            "    Returns:\n"
            "        A list of lists, where each inner list contains words that are anagrams\n"
            "        of each other. The order of groups and words within groups doesn't matter.\n"
            "    \n"
            "    Examples:\n"
            '        >>> group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])\n'
            "        [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]\n"
            '        >>> group_anagrams([""])\n'
            "        [['']]\n"
            '        >>> group_anagrams(["a"])\n'
            "        [['a']]\n"
            '    """\n'
            "```"
        )

        examples = [
            (
                "Input:\n"
                "```python\n"
                "def two_sum(nums: list[int], target: int) -> list[int]:\n"
                '    """\n'
                "    Find indices of two numbers that add up to target.\n"
                "    \n"
                "    Args:\n"
                "        nums: List of integers\n"
                "        target: Target sum\n"
                "    \n"
                "    Returns:\n"
                "        List of two indices [i, j] where nums[i] + nums[j] == target\n"
                "    \n"
                "    Example:\n"
                "        >>> two_sum([2, 7, 11, 15], 9)\n"
                "        [0, 1]\n"
                '    """\n'
                "```\n\n"
                "Output:\n"
                "```python\n"
                "def two_sum(nums: list[int], target: int) -> list[int]:\n"
                "    seen = {}  # value -> index mapping\n"
                "    for i, num in enumerate(nums):\n"
                "        complement = target - num\n"
                "        if complement in seen:\n"
                "            return [seen[complement], i]\n"
                "        seen[num] = i\n"
                "    return []  # No solution found\n"
                "```"
            ),
            (
                "Input:\n"
                "```python\n"
                "def is_palindrome(s: str) -> bool:\n"
                '    """\n'
                "    Check if a string is a palindrome, considering only alphanumeric characters.\n"
                "    Case-insensitive comparison.\n"
                "    \n"
                "    Examples:\n"
                '        >>> is_palindrome("A man, a plan, a canal: Panama")\n'
                "        True\n"
                '        >>> is_palindrome("race a car")\n'
                "        False\n"
                '    """\n'
                "```\n\n"
                "Output:\n"
                "```python\n"
                "def is_palindrome(s: str) -> bool:\n"
                "    # Filter only alphanumeric and convert to lowercase\n"
                "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
                "    # Compare with reversed string\n"
                "    return cleaned == cleaned[::-1]\n"
                "```"
            ),
        ]

        task_name = "code_generation"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 5. READING COMPREHENSION
# Dataset: SQuAD 2.0, QuAC, CoQA
# Evaluation: Extraction accuracy, no hallucination, unanswerable question detection
# =============================================================================
class ReadingComprehension(Task):
    """Class for evaluating reading comprehension capability of LLMs."""

    def __init__(self):
        task_description = (
            "Read the following passage and answer the question based ONLY on the "
            "information provided. If the answer cannot be found in the passage, "
            "respond with 'The passage does not provide this information.'\n\n"
            "PASSAGE:\n"
            "The Great Fire of London began on September 2, 1666, in a bakery on "
            "Pudding Lane owned by Thomas Farriner. The fire spread rapidly due to "
            "strong easterly winds and the predominance of wooden buildings. Over "
            "four days, it destroyed 13,200 houses, 87 churches, and most of the "
            "buildings of the City authorities. Remarkably, the official death toll "
            "was recorded as only six people, though historians believe the actual "
            "number was likely higher as deaths among the poor and middle class "
            "were not always recorded. After the fire, architect Christopher Wren "
            "was commissioned to redesign many buildings, including St. Paul's "
            "Cathedral, which took 35 years to complete.\n\n"
            "QUESTION: How many years did it take to rebuild St. Paul's Cathedral, "
            "and who was the chief architect?"
        )

        examples = [
            (
                "Input:\n"
                "PASSAGE: Marie Curie was born Maria Sklodowska in Warsaw, Poland, "
                "in 1867. She moved to Paris in 1891 to study at the Sorbonne, where "
                "she met her future husband Pierre Curie. Together, they discovered "
                "two elements: polonium (named after Marie's homeland) and radium. "
                "Marie Curie became the first woman to win a Nobel Prize in 1903, "
                "sharing it with Pierre and Henri Becquerel for their work on "
                "radioactivity. She won a second Nobel Prize in 1911, this time in "
                "Chemistry, making her the first person to win Nobel Prizes in two "
                "different scientific fields.\n\n"
                "QUESTION: Why was the element polonium given that name?\n\n"
                "Output: Polonium was named after Poland, Marie Curie's homeland."
            ),
            (
                "Input:\n"
                "PASSAGE: The Amazon rainforest spans approximately 5.5 million square "
                "kilometers, with about 60% located in Brazil. The forest produces "
                "roughly 20% of the world's oxygen through photosynthesis. It is home "
                "to an estimated 10% of all species on Earth, including over 40,000 "
                "plant species and 1,300 bird species.\n\n"
                "QUESTION: What percentage of the Amazon rainforest is located in Peru?\n\n"
                "Output: The passage does not provide this information. It only mentions "
                "that about 60% of the Amazon is in Brazil, but does not specify the "
                "percentage in Peru or any other country."
            ),
        ]

        task_name = "reading_comprehension"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 6. COMMON SENSE REASONING
# Dataset: HellaSwag, PIQA, CommonsenseQA
# Evaluation: Plausible outcome selection, physical causality, everyday knowledge
# =============================================================================
class CommonSenseReasoning(Task):
    """Class for evaluating common sense reasoning capability of LLMs."""

    def __init__(self):
        task_description = (
            "Choose the most logical and plausible completion for the following scenario:\n\n"
            "Sarah put a pot of water on the stove and turned the heat to high. "
            "After 10 minutes, she noticed small bubbles forming at the bottom of the pot. "
            "She then added pasta to the pot.\n\n"
            "What most likely happened next?\n\n"
            "A) The water immediately turned into ice.\n"
            "B) The pasta began to soften and cook as the water continued to heat.\n"
            "C) Sarah removed the pot from the stove because the water was too cold.\n"
            "D) The pot floated up to the ceiling.\n\n"
            "Provide your answer and explain why this is the most plausible outcome."
        )

        examples = [
            (
                "Input: A man is walking his dog in the park on a sunny afternoon. "
                "Suddenly, dark clouds roll in and he hears thunder.\n\n"
                "What will the man most likely do?\n"
                "A) Start sunbathing\n"
                "B) Head home or seek shelter\n"
                "C) Go swimming in the pond\n"
                "D) Remove his umbrella and throw it away\n\n"
                "Output: B) Head home or seek shelter.\n"
                "Reasoning: Thunder indicates an approaching storm with likely rain. "
                "Common sense tells us that people seek shelter when storms approach "
                "to avoid getting wet and to protect themselves from potential lightning. "
                "This is a natural, instinctive response to threatening weather conditions."
            ),
            (
                "Input: Jenny is baking cookies. She places the raw cookie dough on "
                "a baking sheet and puts it in a preheated oven at 350°F.\n\n"
                "What will happen to the cookie dough after 12 minutes?\n"
                "A) The dough will turn into bread\n"
                "B) The cookies will bake and become golden brown\n"
                "C) The dough will freeze solid\n"
                "D) The dough will turn into soup\n\n"
                "Output: B) The cookies will bake and become golden brown.\n"
                "Reasoning: Heat from the oven causes the dough to undergo chemical "
                "changes - the butter melts, sugars caramelize, and the proteins set. "
                "This is standard baking knowledge. Ovens heat food (not freeze it), "
                "and cookie dough bakes into cookies, not bread or soup."
            ),
        ]

        task_name = "common_sense_reasoning"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 7. LANGUAGE UNDERSTANDING & AMBIGUITY
# Dataset: WinoGrande, WSC (Winograd Schema Challenge)
# Evaluation: Correct pronoun resolution, semantic nuance understanding
# =============================================================================
class LanguageUnderstanding(Task):
    """Class for evaluating language understanding and ambiguity resolution."""

    def __init__(self):
        task_description = (
            "Read the following sentences and determine what the pronoun refers to. "
            "Explain your reasoning.\n\n"
            "Sentence 1: 'The trophy didn't fit in the suitcase because it was too big.'\n"
            "Question: What was too big - the trophy or the suitcase?\n\n"
            "Sentence 2: 'The trophy didn't fit in the suitcase because it was too small.'\n"
            "Question: What was too small - the trophy or the suitcase?\n\n"
            "Answer both questions and explain how changing one word completely changes "
            "the reference of 'it'."
        )

        examples = [
            (
                "Input:\n"
                "Sentence: 'The lawyer questioned the witness because he was lying.'\n"
                "Question: Who was lying - the lawyer or the witness?\n\n"
                "Output: The witness was lying.\n"
                "Reasoning: In typical legal contexts, lawyers question witnesses to "
                "expose lies or inconsistencies. It would be unusual for a lawyer to "
                "question someone because the lawyer himself was lying. The causal "
                "relationship ('because') makes sense only if 'he' refers to the witness - "
                "the lawyer questioned the witness in order to expose the witness's lies."
            ),
            (
                "Input:\n"
                "Sentence: 'The mothers of the football players make their meals.'\n"
                "Question: Whose meals do the mothers make - the mothers' own meals or "
                "the football players' meals?\n\n"
                "Output: The mothers make the football players' meals.\n"
                "Reasoning: The phrase 'their meals' most naturally refers to the football "
                "players because: (1) the sentence establishes a relationship between mothers "
                "and players, (2) mothers commonly prepare food for their children, and "
                "(3) saying mothers make 'their own meals' would be unremarkable and odd "
                "to mention specifically in relation to football players."
            ),
        ]

        task_name = "language_understanding"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 8. FACTUAL KNOWLEDGE & RETRIEVAL
# Dataset: MMLU, TriviaQA, Natural Questions
# Evaluation: Factual accuracy, precision, no confabulation
# =============================================================================
class FactualKnowledge(Task):
    """Class for evaluating factual knowledge and retrieval capability."""

    def __init__(self):
        task_description = (
            "Answer the following questions from different knowledge domains. "
            "Be precise and provide only factually correct information. "
            "If you're not certain, indicate your uncertainty.\n\n"
            "1. HISTORY: In what year did the Berlin Wall fall, and which two "
            "countries did it separate?\n\n"
            "2. SCIENCE: What is the chemical symbol for gold, and what is its "
            "atomic number?\n\n"
            "3. GEOGRAPHY: What is the longest river in South America, and "
            "approximately how long is it in kilometers?\n\n"
            "Provide concise, accurate answers for each question."
        )

        examples = [
            (
                "Input:\n"
                "1. Who wrote the novel '1984' and when was it published?\n"
                "2. What is the speed of light in a vacuum (in km/s)?\n"
                "3. What is the capital city of Australia?\n\n"
                "Output:\n"
                "1. LITERATURE: '1984' was written by George Orwell (pen name of "
                "Eric Arthur Blair) and published in 1949.\n\n"
                "2. PHYSICS: The speed of light in a vacuum is approximately "
                "299,792 kilometers per second (often rounded to 300,000 km/s).\n\n"
                "3. GEOGRAPHY: The capital city of Australia is Canberra (not Sydney "
                "or Melbourne, which are larger cities but not the capital)."
            ),
            (
                "Input:\n"
                "What element has the highest melting point, and what is that temperature?\n\n"
                "Output:\n"
                "Tungsten (chemical symbol W) has the highest melting point of all "
                "elements at approximately 3,422°C (6,192°F). This property makes it "
                "ideal for use in incandescent light bulb filaments and high-temperature "
                "industrial applications."
            ),
        ]

        task_name = "factual_knowledge"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 9. MATHEMATICAL PROBLEM SOLVING
# Dataset: GSM8K, MATH, AQuA
# Evaluation: Correct operations, multi-step reasoning, accurate final answer
# =============================================================================
class MathematicalProblemSolving(Task):
    """Class for evaluating mathematical problem solving capability."""

    def __init__(self):
        task_description = (
            "Solve the following word problem. Show your work step by step.\n\n"
            "Problem: A bookstore had a sale where hardcover books were discounted "
            "by 25% and paperbacks were discounted by 15%. Tom bought 3 hardcover "
            "books originally priced at $24 each and 5 paperback books originally "
            "priced at $12 each. If there is an 8% sales tax applied after all "
            "discounts, how much did Tom pay in total?\n\n"
            "Provide your solution with clear steps and the final answer."
        )

        examples = [
            (
                "Input: A bakery sells cupcakes in boxes of 6 and cookies in boxes of 8. "
                "If someone buys 3 boxes of cupcakes and 4 boxes of cookies, "
                "how many individual items did they buy in total?\n\n"
                "Output:\n"
                "Step 1: Calculate total cupcakes\n"
                "Number of cupcakes = 3 boxes × 6 cupcakes/box = 18 cupcakes\n\n"
                "Step 2: Calculate total cookies\n"
                "Number of cookies = 4 boxes × 8 cookies/box = 32 cookies\n\n"
                "Step 3: Calculate total items\n"
                "Total items = 18 cupcakes + 32 cookies = 50 items\n\n"
                "Final Answer: They bought 50 individual items in total."
            ),
            (
                "Input: A farmer has 5 chicken coops. Each coop has 12 chickens. "
                "If each chicken lays 2 eggs per day, how many eggs does the farmer "
                "collect in one week?\n\n"
                "Output:\n"
                "Step 1: Find the total number of chickens\n"
                "Total chickens = 5 coops × 12 chickens/coop = 60 chickens\n\n"
                "Step 2: Calculate eggs per day\n"
                "Eggs per day = 60 chickens × 2 eggs/chicken = 120 eggs\n\n"
                "Step 3: Calculate eggs per week\n"
                "Eggs per week = 120 eggs/day × 7 days = 840 eggs\n\n"
                "Final Answer: The farmer collects 840 eggs in one week."
            ),
        ]

        task_name = "mathematical_problem_solving"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


# =============================================================================
# 10. ETHICAL REASONING & NUANCE
# Dataset: ETHICS (Hendrycks et al.), Moral Machine, Do Not Answer
# Evaluation: Balanced analysis, multiple perspectives, ethical frameworks
# =============================================================================
class EthicalReasoning(Task):
    """Class for evaluating ethical reasoning capability."""

    def __init__(self):
        task_description = (
            "Analyze the following ethical dilemma from multiple perspectives.\n\n"
            "Scenario: A self-driving car's AI system must make a split-second decision. "
            "The brakes have failed, and the car is heading toward a crosswalk with "
            "three elderly pedestrians. The only alternative is to swerve into a "
            "barrier, which would likely kill the single passenger in the car. "
            "The AI must choose between these two outcomes.\n\n"
            "Discuss this dilemma considering:\n"
            "1. A utilitarian perspective (greatest good for greatest number)\n"
            "2. A deontological perspective (duty-based ethics)\n"
            "3. What practical and societal implications such programming decisions might have\n\n"
            "Provide a balanced analysis without advocating for a single 'correct' answer."
        )

        examples = [
            (
                "Input: A doctor has five patients who will die without organ transplants. "
                "A healthy patient comes in for a routine checkup. Should the doctor "
                "sacrifice the one healthy patient to save the five dying patients?\n\n"
                "Output:\n"
                "This is a classic ethical thought experiment that highlights tensions "
                "between different moral frameworks.\n\n"
                "UTILITARIAN VIEW: A purely consequentialist calculation might suggest "
                "that saving five lives at the cost of one produces more total well-being. "
                "However, most utilitarian philosophers would reject this action because "
                "of the broader consequences: if doctors could sacrifice patients, trust "
                "in medicine would collapse, causing far greater harm.\n\n"
                "DEONTOLOGICAL VIEW: From a duty-based perspective, the action is clearly "
                "wrong. The doctor has a fundamental duty not to harm patients. Using a "
                "person merely as a means to an end (saving others) violates their inherent "
                "dignity and autonomy. The doctor's duty to 'do no harm' is inviolable.\n\n"
                "VIRTUE ETHICS: A virtuous doctor would not perform this action. The virtues "
                "of compassion, integrity, and trustworthiness all counsel against it.\n\n"
                "CONCLUSION: While the thought experiment creates an artificial tension, "
                "real ethical reasoning recognizes that some actions are impermissible "
                "regardless of outcomes. The near-universal moral intuition against this "
                "action reflects deeply held principles about human dignity."
            ),
            (
                "Input: Is it ethical to lie to protect someone's feelings?\n\n"
                "Output:\n"
                "This question involves weighing honesty against kindness—both important "
                "values that can sometimes conflict.\n\n"
                "ARGUMENTS FOR 'WHITE LIES':\n"
                "- Protecting someone from unnecessary pain shows compassion\n"
                "- Social harmony often depends on tactful communication\n"
                "- The intent is benevolent, not self-serving\n\n"
                "ARGUMENTS FOR STRICT HONESTY:\n"
                "- Deception, even well-intentioned, undermines trust\n"
                "- People have a right to accurate information to make decisions\n"
                "- Lies can compound and cause greater harm later\n\n"
                "NUANCED VIEW: Context matters significantly. A surprise party requires "
                "temporary deception. Telling someone their terminal diagnosis differs "
                "from commenting on their new haircut. The relationship, stakes, and "
                "alternatives all factor into ethical evaluation.\n\n"
                "Most ethical frameworks would advise: prefer honest but kind communication, "
                "avoid gratuitous honesty that serves no purpose, and recognize that "
                "respecting someone sometimes means giving them truth they need to hear."
            ),
        ]

        task_name = "ethical_reasoning"
        super().__init__(
            task_description=task_description, examples=examples, task_name=task_name
        )


if __name__ == "__main__":
    ftsk = InstructionsFollowing()
    print(ftsk.task_description)
    print(ftsk.cot_prompt)
    print(ftsk.few_shot_prompt)
