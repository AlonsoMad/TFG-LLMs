# NOTE: When implementing MIND, we can think of keeping each component independent (one file for each component), or all at once (one file for all components). For later fine-tuning, it may make more sense to keep them in different files, but I let you decide.

######################
# PATHS TO TEMPLATES #
######################
# SEARCH QUERY GENERATION
import logging
import pathlib
from src.prompter.prompter import Prompter
from src.utils.utils import init_logger

class QueryGenerator():
    def __init__(
        self,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("config/config.yaml"),
        instructions_path: pathlib.Path = pathlib.Path("src/mind/templates/query_generation.txt"),
        model_type: str = "llama3.3:70b",  # you can try other models; you can check available models by doing `ollama list` in the terminal (this only works from kumo01)
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
        # TODO: I'd probably load default config here with instruction_path, default model_type, etc. rather than passing each independently, plus var args that could be passed
        with open(instructions_path, 'r') as file: self.template = file.read()
        self.model_type = model_type # it may make more sense to make this an argument that you pass when making the generate_query call, rather than an attribute of the class (in case we want to test different LLMs, which we will)

        self.prompter = Prompter(
            model_type=self.model_type,
            # TODO: at this point not relevant, but a a later point you will need to experiment with these (e.g. temperature, max_tokens, etc.)
        )
        
        self._logger.info(f"Prompter initialized")

    def generate_query(self, question, passage):
        """Generates search queries from a question; the passage is used as 'context' for the LLM (it is assumed that the question has been generated from the passage)"""
        template_formatted = self.template.format(question=question, passage=passage)

        sq, _ = self.prompter.prompt(question=template_formatted)
        # parse the search queries
        try:
            sq_clean = [el for el in sq.split(";") if el.strip()]
        except Exception as e:
            print(f"******Error extracting queries: {e}")
            
        return sq_clean