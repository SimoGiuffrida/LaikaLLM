from abc import ABC, abstractmethod
import random

from requests.structures import CaseInsensitiveDict


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
    # name obj class mapping, used for when task must be initialized from strings
    str_alias_obj: dict = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_obj dict
    def __init_subclass__(cls, **kwargs):
        cls.str_alias_obj[cls.__name__] = cls

    @property
    def all_templates(self):
        return list(self.templates_dict.values())

    def force_template(self, force_template_id: int):

        # 'self.__class__' so that even if we call multiple times this method on an instantiated task,
        # we always have a pointer to original class templates, otherwise they are deleted if we use simply 'self'
        # instead of self.__class__

        if force_template_id not in set(self.__class__.templates_dict.keys()):
            raise KeyError(f"Prompt template id {force_template_id} not found! "
                           f"Available prompt ids are {list(self.templates_dict.keys())}")

        self.templates_dict = {force_template_id: self.__class__.templates_dict[force_template_id]}

        return self

    # function decorator needed to declare mandatory arguments of each subclass __call__
    @staticmethod
    def validate_args(*mandatory_args: str):
        def decorator(func):
            def wrapper(self, **kwargs):
                for mandatory_arg in mandatory_args:
                    assert mandatory_arg in kwargs, f"{mandatory_arg} is needed for task {repr(self)}!"

                return func(self, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def from_string(cls, *task_str: str):

        try:
            # remember, we are searching a case-insensitive dict, so we don't care about
            # lowering all keys
            instantiated_task = [cls.str_alias_obj[task]() for task in task_str]
        except KeyError:
            raise KeyError("One or more task string alias does not exist!") from None

        return instantiated_task

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)


class SequentialTask(Task):
    templates_dict = {
        0: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Predict for the user the next element of the following sequence -> \n"
                         "{}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Predict the next element which the user will buy given the following order history -> \n"
                         "{}",
            target_text="{}"
        ),
        2: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "What is the element that should be recommended to the user knowing that it has bought -> \n"
                         "{}",
            target_text="{}"
        ),
        3: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Recommend to the user an item from the catalog given its order history -> \n"
                         "{}",
            target_text="{}"
        ),
        4: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "This is the order history of the user -> \n"
                         "{} \n"
                         "Recommend the next element that the user will buy",
            target_text="{}"
        ),
        5: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Please predict what item is best to recommend to the user given its order history -> \n"
                         "{}",
            target_text="{}"
        )
    }

    @Task.validate_args("user_id", "input_item_seq", "target_item")
    def __call__(self, **kwargs):
        user_id = kwargs["user_id"]
        order_history = kwargs["input_item_seq"]
        target_item = kwargs["target_item"]

        # random.choice applied to dict with int key returns a value
        input_prompt, target = random.choice(self.all_templates)

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        order_history_str = separator.join(order_history)

        input_text = input_prompt.format(user_id, order_history_str)
        target_text = target.format(target_item)

        return input_text, target_text
