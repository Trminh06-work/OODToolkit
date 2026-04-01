from .benchmark import AnalystModel
from .splitters import BaseSplitter
from typing import List

def main_split(splitters: List[BaseSplitter]):
    for splitter in splitters:
        splitter.split()


def main_train():
    pass


def main_eval():
    pass


def main(splitters: List[BaseSplitter] = None, require_train = False, require_eval = False):
    if splitters is not None:
        main_split(splitters)
    if require_train:
        main_train()
    if require_eval:
        main_eval()


if __name__ == "__main__":
    main()