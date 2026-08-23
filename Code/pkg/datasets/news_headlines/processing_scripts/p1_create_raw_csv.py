"""
Build OriginalData.csv for the news_headlines dataset from the two raw JSON-lines files.

Both v1 (26,709 rows) and v2 (28,619 rows) are Misra's Sarcasm Headlines corpus; v2 is a
later, larger collection that overlaps v1 heavily. We merge them and de-duplicate on the
headline text, which yields more training data than either file alone.

`article_link` is dropped rather than carried through: every sarcastic headline comes from
theonion.com and every non-sarcastic one from huffingtonpost.com, so the URL *is* the
label. Keeping it would leak the target.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / 'raw_data'
PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'processed_data'

RAW_FILES = ['Sarcasm_Headlines_Dataset.json', 'Sarcasm_Headlines_Dataset_v2.json']


def load_raw() -> pd.DataFrame:
    """
    Read every raw JSON-lines file and return a single frame of unique headlines.

    The two files store the same three keys in a different column order, so columns are
    selected by name. The original version of this script passed `header=[...]` to
    to_csv(), which renames positionally -- on v1, whose order is
    (article_link, headline, is_sarcastic), that silently wrote URLs into the
    'sarcasm_label' column.
    """
    frames = []
    for file_name in RAW_FILES:
        path = RAW_DIR / file_name
        if not path.is_file():
            raise FileNotFoundError(
                'Missing raw dataset "' + str(path) + '" - see the README for download instructions')
        frame = pd.read_json(path, lines=True)
        print('Read ' + str(len(frame)) + ' rows from ' + file_name)
        frames.append(frame[['is_sarcastic', 'headline']])

    data_frame = pd.concat(frames, ignore_index=True)
    data_frame = data_frame.rename(columns={'is_sarcastic': 'sarcasm_label', 'headline': 'text_data'})

    # Strip whitespace before de-duplicating, otherwise "a headline" and "a headline "
    # survive as two rows and can land on opposite sides of the train/test split.
    data_frame['text_data'] = data_frame['text_data'].astype(str).str.strip()
    data_frame = data_frame[data_frame['text_data'].str.len() > 0]
    data_frame = data_frame.drop_duplicates(subset='text_data', keep='first').reset_index(drop=True)
    return data_frame


if __name__ == '__main__':
    data = load_raw()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / 'OriginalData.csv'
    data.to_csv(path_or_buf=output_path, index=False)

    label_counts = data['sarcasm_label'].value_counts().sort_index()
    print('\nWrote ' + str(len(data)) + ' unique headlines to ' + str(output_path))
    print('  non-sarcastic (0): ' + str(label_counts.get(0, 0)))
    print('  sarcastic     (1): ' + str(label_counts.get(1, 0)))
