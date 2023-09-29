from abc import ABC, abstractmethod
import random
from collections import namedtuple
from typing import List, NamedTuple


class PromptTarget:

    def __init__(self, input_prompt: str, target_text: str):
        self.input_prompt = input_prompt
        self.target_text = target_text

    # iter just so that this class can be unpacked,
    # e.g. input_prompt, target_text = PromptTarget(...)
    def __iter__(self):
        return iter((self.input_prompt, self.target_text))

    def __str__(self):
        string = " Input ".center(50, "#") + "\n"
        string += self.input_prompt + "\n"
        string += " Target ".center(50, "#") + "\n"
        string += self.target_text

        return string


class Task(ABC):
    # keys are integers, values are PromptTarget objects
    templates_dict = {}

    def __init__(self, force_template_id: int = None):
        if force_template_id is not None:
            try:
                self.templates_dict = {force_template_id: self.templates_dict[force_template_id]}
            except KeyError:
                raise KeyError(f"Prompt template id {force_template_id} not found! "
                               f"Available prompt ids are {list(self.templates_dict.keys())}") from None

        self.all_templates = list(self.templates_dict.values())

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class SequentialTask(Task):
    templates_dict = {
        0: PromptTarget(
            input_prompt="sequential_rec:\n\n"
                         "Predict the next element of the following sequence for {} ->\n"
                         "{}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="sequential_rec:\n\n"
                         "Predict the next element that {} will buy given the following order history ->\n"
                         "{}",
            target_text="{}"
        ),
        2: PromptTarget(
            input_prompt="sequential_rec:\n\n"
                         "What is the element that should be recommended to {} knowing that it has bought ->\n"
                         "{}",
            target_text="{}"
        ),
        3: PromptTarget(
            input_prompt="sequential_rec:\n\n"
                         "Recommend to {} an item from the catalog given its order history ->"
                         "{}",
            target_text="{}"
        ),
        4: PromptTarget(
            input_prompt="sequential_rec:\n\n"
                         "This is the order history of {} ->\n"
                         "{}\n\n"
                         "Recommend the next element that the user will buy",
            target_text="{}"
        ),
        5: PromptTarget(
            input_prompt="sequential_rec:\n\n"
                         "Please predict, for {}, what item is best to recommend given its order history ->\n"
                         "{}",
            target_text="{}"
        )
    }

    def __call__(self, user_id: str, order_history: List[str], target_item_id: str):

        # random.choice applied to dict with int key returns a value
        input_prompt, target = random.choice(self.all_templates)

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        order_history_str = separator.join(order_history)

        input_text = input_prompt.format(user_id, order_history_str)
        target_text = target.format(target_item_id)

        return input_text, target_text
