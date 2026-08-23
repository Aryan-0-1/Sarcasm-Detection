"""
Interactive command-line sarcasm classifier.

    python Code/console.py

Prints a score and class label per input, and refreshes colorise.html with the attention
heat-map -- same output contract as the 2020 version, but backed by the fine-tuned
DistilBERT model instead of the ELMo BiLSTM.
"""
print('Starting Console...')
import sys
import pathlib; base_path = pathlib.Path(__file__).parent.parent.resolve(); sys.path.insert(1, str(base_path))

from Code.pkg.analysis.attention_html import render_attention_html, write_attention_html
from Code.pkg.model_training.transformer import load_model, predict

VISUALISATION_PATH = pathlib.Path(__file__).parent / 'colorise.html'


def main():
    try:
        tokenizer, model, resolved_id = load_model()
    except FileNotFoundError as error:
        print('\n' + str(error))
        return 1

    print('Model: ' + resolved_id)

    while True:
        sentence = ''
        while not sentence.strip():
            sentence = input('\nEnter Text: ')

        result = predict(sentence, tokenizer, model)

        html = render_attention_html(result['tokens'], result['weights'], result['score'])
        write_attention_html(str(VISUALISATION_PATH), html)

        print('\nPrediction score: ' + str(round(result['score'], 4)))
        print('Class label: ' + result['label'])
        print('A visualisation is now available at ' + str(VISUALISATION_PATH))

        content = ''
        while content not in {'y', 'n'}:
            content = input('\nContinue? y / n \n').strip().lower()
            if content not in {'y', 'n'}:
                print('Invalid input - press "y" to continue, or "n" to exit')

        if content == 'n':
            return 0


if __name__ == "__main__":
    sys.exit(main())
