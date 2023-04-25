import srt
import os


def end_of_sentence(sen, stop=('...', '.', '?', '!', '!!!')):
    """Check if sentence ends with a stop character."""
    return sen.endswith(stop)


def extract_sentences(file):
    """Extract sentences from subtitle file."""
    sentances = []
    sub_composed = ""
    for sub in srt.parse(file):
        sub_ = sub.content.replace("\n", " ").lstrip()
        sub_ = sub_.replace("<i>", "").replace("</i>", "")

        sub_composed += " " + sub_
        if end_of_sentence(sub_composed):
            sentances.append(sub_composed)
            sub_composed = ""

    return sentances


def save_sentences(s, f_name):
    """Save sentences to file."""
    with open(f_name, "w") as file:
        for sen in s:
            file.write(sen + "\n")


def load_sentences(f_name):
    """Load sentences from file."""
    with open(f_name, "r") as file:
        return [sen[:-1] for sen in file.readlines()]


if __name__ == "__main__":
    subtitles_dir = "./subtitles/"
    for filename in os.scandir(subtitles_dir):
        if filename.is_file() and filename.path.endswith(".srt"):
            print(filename.path)
            f = open(filename.path, encoding="Windows-1252").read()
            f = f.replace("è", "č").replace("È", "Č")
            sen = extract_sentences(f)

            f_name = filename.path.split("/")[-1].split(".")[0]
            save_sentences(sen, f_name=f"./sentences/{f_name}_sen.txt")
