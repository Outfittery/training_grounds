import pandas as pd

if __name__ == '__main__':
    df = pd.read_parquet('lstm_task.parquet')
    df.index.name='sentence_id'
    df.columns = [c.replace('letter_','word_') for c in df.columns]
    df.to_parquet('lstm_task.parquet')