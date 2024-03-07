from typing import *
import os
import struct
from pathlib import Path
from yo_fluq_ds import Query, FileIO
import pandas as pd
import numpy as np

class GloveProcessor:
    def __init__(self, build_path, test=False):
        self.build_path = build_path

    def create_vocab(self, corpus_file: Path, vocab_file: Path, min_count = 0, verbose = 2, debug=False):
        cmd = f'{self.build_path/"vocab_count"} -min-count {min_count} -verbose {verbose} < {corpus_file} > {vocab_file}'
        if not debug:
            return os.system(cmd)
        else:
            return cmd

    def create_cooccur(self, corpus_file: Path, vocab_file: Path, cooc_file: Path, window_size=10, memory='4.0', verbose=2, debug=False):
        cmd = (
            f'{self.build_path/"cooccur"} -memory {memory} -vocab-file {vocab_file} -verbose {verbose}'
            f'-window-size {window_size} < {corpus_file} > {cooc_file}'
        )
        if not debug:
            return os.system(cmd)
        else:
            return cmd


    def shuffle_cooccur(self, source_cooc: Path, result_cooc: Path, memory = '4.0', verbose=2, debug=False):
        cmd = f'{self.build_path/"shuffle"} -memory {memory} -verbose {verbose} < {source_cooc} > {result_cooc}'

        if not debug:
            return os.system(cmd)
        else:
            return cmd


    def execute_glove(self,
                      vocab_file: Path,
                      cooc_file: Path,
                      output: Path,
                      threads=1,
                      iterations = 10,
                      vector_size=50,
                      binary = 2,
                      xmax=10,
                      verbose=2,
                      debug=False,
                      model=1):
        cmd = (
            f'{self.build_path}/glove -save-file {output} -threads {threads} -input-file {cooc_file} '
            f'-x-max {xmax} -iter {iterations} -vector-size {vector_size} -binary {binary} -vocab-file {vocab_file} '
            f'-verbose {verbose} -model {model}'
        )

        if not debug:
            return os.system(cmd)
        else:
            return cmd


    @staticmethod
    def read_glove_scores(file: Path, word_to_index_series: Optional[pd.Series] = None):
        query = (
            Query
                .file.text(file)
                .select(lambda z: z.split(' '))
                .select(lambda z: [x if ind == 0 else float(x) for ind, x in enumerate(z)])
        )
        df = query.to_dataframe()
        cols = list(df.columns)
        cols[0] = 'word'
        df.columns = cols
        df = df.set_index('word')

        if word_to_index_series is not None:
            df = df.loc[df.index != '<unk>']
            name = word_to_index_series.name
            df = df.merge(word_to_index_series.to_frame(name), left_index=True, right_index=True)
            df = df.set_index(name)


        return df

    @staticmethod
    def apply_scores(
            x: pd.Series,
            y: pd.Series,
            scores: pd.DataFrame,
            unk_key: Optional[str] = None,
            bias_term: bool = False,
            context_vector: bool = False,
    ):
        if unk_key is not None:
            x = pd.Series(np.where(x.isin(scores.index), x, unk_key), x.index)
            y = pd.Series(np.where(y.isin(scores.index), y, unk_key), y.index)
        scores_x = scores_y = scores
        if context_vector:
            vector_end = scores.shape[1] // 2
            scores_x = scores_x.iloc[:, :vector_end]
            scores_y = scores_y.iloc[:, vector_end:]
            scores_y.columns = scores_x.columns
        mx = x.to_frame('idx').merge(scores_x, left_on='idx', right_index=True, how='left').drop('idx', axis=1).sort_index()
        my = y.to_frame('idx').merge(scores_y, left_on='idx', right_index=True, how='left').drop('idx', axis=1).sort_index()
        bias_x = bias_y = 0
        if bias_term:
            bias_x = mx.iloc[:, -1]
            bias_y = my.iloc[:, -1]
            mx = mx.iloc[:, :-1]
            my = my.iloc[:, :-1]
        m = mx * my
        val = m.sum(axis=1) + bias_x + bias_y
        isn = m.isnull().any(axis=1)
        if isn.any():
            if unk_key is not None:
                raise ValueError('Something strange has happened: when unk_key is provided, no Nones are expected in the output')
            else:
                val.loc[isn] = None
        return val

    @staticmethod
    def read_cooc_iter(cooc_file: Path):
        with open(cooc_file,'rb') as file:
            while(True):
                data = file.read(16)
                if len(data)==0:
                    break
                yield struct.unpack('iid', data)

    @staticmethod
    def read_cooc(cooc_file: Path):
        return Query.en(GloveProcessor.read_cooc_iter(cooc_file))

    @staticmethod
    def _write_one_value(word_0, word_1, value, file, add_symmetric = False):
        file.write(struct.pack('iid', word_0 + 1, word_1 + 1, value))
        if add_symmetric:
            if word_0!=word_1:
                file.write(struct.pack('iid', word_1 + 1, word_0 + 1, value))

    @staticmethod
    def write_cooc(data: Iterable[Tuple[int,int,int]], file, add_symmetric=False):
        with open(file, 'wb') as file:
            for word0, word1, value in data:
                GloveProcessor._write_one_value(word0, word1, value, file, add_symmetric)

    @staticmethod
    def write_vocab(word_to_id, id_to_count, file: Path):
        id_to_word = {ind:word for word,ind in word_to_id.items()}

        rows = []
        for i in range(len(id_to_word)):
            if i not in id_to_word:
                raise ValueError('Non-continuous id_to_word!')
            rows.append(f'{id_to_word[i]} {id_to_count.get(i, 0)}')

        FileIO.write_text('\n'.join(rows), file)






