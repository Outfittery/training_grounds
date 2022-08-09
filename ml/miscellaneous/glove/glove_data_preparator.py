from ...._common import DataBundle
from pathlib import Path
from .glove_processor import GloveProcessor
import pandas as pd
from yo_fluq_ds import Query, Queryable, fluq
import numpy as np



class GloveDataPreparator:
    @staticmethod
    def new(df, target_column, group_column):
        db = DataBundle(src=df)
        db.additional_information.target_column = target_column
        db.additional_information.group_column = group_column
        return GloveDataPreparator(db)

    def __init__(self, db):
        self.db = db
        self.df = db.src
        self.target_column = db.additional_information.target_column
        self.group_column = db.additional_information.group_column

    def build_index(self):
        idx = self.df[self.target_column].drop_duplicates()
        idx = idx.reset_index(drop=True).reset_index(drop=False)
        idx = idx.set_index(self.target_column)['index'].to_frame('idx')
        self.db['word_to_index'] = idx

    def fixed_df(self):
        ids = list(self.db.word_to_index.loc[self.df[self.target_column]]['idx'])
        df = pd.DataFrame(dict(
            id=list(ids),
            group=list(self.df[self.group_column])
        ))
        return df

    def build_pairs(self, remove_diagonal=True):
        df = self.fixed_df()
        df = df.merge(df.set_index('group')[['id']], left_on='group', right_index=True)
        self.db['pairs'] = df.groupby(['id_x', 'id_y']).size().to_frame('cnt').reset_index()

    def build_all(self):
        self.build_index()
        self.build_pairs()

    def generate_vocab_file(self, path: Path, glove_processor: GloveProcessor):
        word_to_id = self.db.word_to_index.idx.to_dict()
        id_to_counts = self.fixed_df().groupby('id').size().to_dict()
        glove_processor.write_vocab(word_to_id, id_to_counts, path)

    def generate_cooc_file(self, path, glove_processor: GloveProcessor, with_progress_bar = False):
        src = Query.en(self.db.pairs[['id_x', 'id_y', 'cnt']].iterrows()).select(lambda z: tuple(z[1]))
        src = Queryable(src, self.db.pairs.shape[0])
        if with_progress_bar:
            src = src.feed(fluq.with_progress_bar())
        glove_processor.write_cooc(src, path, add_symmetric=False)

    def read_and_translate_scores(self, path, glove_processor: GloveProcessor, translate=True):
        tr = None
        if translate:
            tr = self.db.word_to_index.idx
        return GloveProcessor.read_glove_scores(path, tr)


    def evaluate(self, x, y, scores):
        mx = x.to_frame('idx').merge(scores, left_on='idx', right_index=True, how='left').drop('idx', axis=1).sort_index()
        my = y.to_frame('idx').merge(scores, left_on='idx', right_index=True, how='left').drop('idx', axis=1).sort_index()
        m = mx*my
        val = m.sum(axis=1)
        isn = m.isnull().any(axis=1)
        val.loc[isn]=None
        return val


    def evaluate_df(self, df, scores):
        cs = list(df.columns)
        return self.evaluate(df[cs[0]], df[cs[1]], scores)

    def _make_batch(self, N):
        ids = list(self.db.word_to_index.idx)
        df = pd.DataFrame(dict(
            id_x=np.random.choice(ids, size=N),
            id_y=np.random.choice(ids, size=N),
        ))
        df = df.loc[df.id_x != df.id_y]
        df = df.merge(self.db.pairs.set_index(['id_x', 'id_y']), left_on=['id_x', 'id_y'], right_index=True, how='left')
        df = df.loc[~df.cnt.isnull()].drop('cnt', axis=1)
        return df

    def generate_negative_samples(self, N):
        result = None
        while True:
            df = self._make_batch(N)
            result = df if result is None else pd.concat([result, df])
            result = result.drop_duplicates()
            if result.shape[0] >= N:
                return result.iloc[:N].reset_index(drop=True)

    @staticmethod
    def assign_factor_and_split(df, nfactor, test_size):
        pdf = df.sort_values('cnt', ascending=False).copy()
        pdf['factor'] = 0
        k = pdf.shape[0]
        for i in range(nfactor):
            pdf.loc[pdf.iloc[:k].index, 'factor'] = i + 1
            k //= 2
        pdf['random'] = np.random.random(size=pdf.shape[0])
        pdf = pdf.feed(fluq.add_ordering_column('factor', 'random'))
        N = test_size
        pdf['split'] = np.where(
            pdf.order.between(0, test_size),
            'test',
            np.where(
                pdf.order.between(test_size, 2 * test_size),
                'display',
                'train'
            )
        )
        return pdf

    def adjust_bundle(self, splitted_pairs, test_df):
        new_pairs = splitted_pairs.loc[splitted_pairs.split != 'test']
        test_df = test_df.assign(cnt=0, factor=0, split='test')
        control = pd.concat([
            splitted_pairs.loc[splitted_pairs.split != 'train'],
            test_df
        ])
        control = control.reset_index(drop=True)

        new_db = self.db.copy()
        new_db['full_pairs'] = pd.concat([splitted_pairs,test_df])
        new_db['pairs'] = new_pairs
        new_db['control'] = control
        return new_db

    def make_train_control(self, nfactor, test_size=5000):
        pdf = GloveDataPreparator.assign_factor_and_split(self.db.pairs, nfactor, test_size)
        neg_sample = self.generate_negative_samples(test_size)
        return self.adjust_bundle(pdf, neg_sample)
