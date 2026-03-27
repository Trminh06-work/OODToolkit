def main_split():
    pass


def main_train():
    pass


def main_eval():
    pass


def main(require_split = False, require_train = False, require_eval = False):
    if require_split:
        main_split()
    if require_train:
        main_train()
    if require_eval:
        main_eval()

if __name__ == "__main__":
    main()