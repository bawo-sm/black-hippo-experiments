from dataclasses import dataclass


@dataclass(frozen=True)
class Prompts:
    describe_image: str

    @classmethod
    def from_text_files(cls, filepaths: list[str]):
        kwargs = {}
        for x in filepaths:
            prompt_name = x.split("/")[-1].split(".")[0]
            kwargs[prompt_name] = x
        return cls(**kwargs)


filepahts = [
    "src/ai/prompts/describe_image.txt"
]
prompts_manager = Prompts.from_text_files(filepahts)
