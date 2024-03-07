from yo_fluq_ds import *
from typing import *
import hashlib
from datetime import datetime
import boto3
from .utils import download_and_open_sagemaker_result
from matplotlib import pyplot as plt
from ...data import CacheMode
from ..._common import Loc

class S3TrainingLogsLoader:
    def __init__(self,
                 bucket: str = None,
                 project_name: str = None,
                 ):
        self.bucket = bucket
        self.project_name = project_name

    def find_jobs(self, from_time: datetime, to_time: Optional[datetime] = None):
        args = dict(CreationTimeAfter=from_time)
        if to_time is not None:
            args['CreationTimeBefore'] = to_time

        client = boto3.client('sagemaker')
        job_ids = [job['TrainingJobName'] for job in client.list_training_jobs(MaxResults=100, **args)['TrainingJobSummaries']]
        return job_ids

    def get_jobs_df(self, from_time: datetime, to_time: Optional[datetime] = None):
        args = dict(CreationTimeAfter=from_time)
        if to_time is not None:
            args['CreationTimeBefore'] = to_time

        client = boto3.client('sagemaker')
        jobs = [job for job in client.list_training_jobs(MaxResults=100, **args)['TrainingJobSummaries']]
        jdf = pd.DataFrame(jobs)
        jdf['duration_in_hours'] = (jdf.TrainingEndTime - jdf.CreationTime).dt.total_seconds() / 3600
        return jdf


    def _load_metrics_df(self, rs):
        df = pd.DataFrame(FileIO.read_pickle(rs.get_path('output/history.pkl')))
        rdf = df.drop('timestamp', axis=1).unstack().to_frame().reset_index()
        rdf.columns = ['metric', 'ordinal', 'value']
        if 'timestamp' in df.columns:
            rdf = rdf.merge(pd.to_datetime(df.timestamp), left_on='ordinal', right_index=True).reset_index(drop=True)
        return rdf

    def load_metrics(self, job_ids, progress_bar = False, ignore_errors = False):
        dfs = []
        job_ids = Query.en(job_ids)
        if progress_bar:
            job_ids = job_ids.feed(fluq.with_progress_bar())
        for job_id in job_ids:
            rs = None
            try:
                rs = download_and_open_sagemaker_result(self.bucket, self.project_name, job_id, dont_redownload=True)
            except Exception as ex:
                if not ignore_errors:
                    raise ValueError(f"Job_id: {job_id}") from ex
            if rs is None:
                continue
            df = self._load_metrics_df(rs)
            df['job_id'] = job_id
            dfs.append(df)
        return pd.concat(dfs)

    def load_and_cache_metrics(self, job_ids, progress_bar = False, ignore_errors = False, cache_mode = 'default', cache_path = None):
        if cache_path is None:
            code = '/'.join([c for job in job_ids for c in job])
            name = hashlib.md5(code.encode('utf-8')).hexdigest()
            cache_name = 'sagemaker-' + name+'.parquet'
            cache_path = Loc.data_cache_path / cache_name
        return CacheMode.from_parquet_file().get(
            cache_path,
            lambda: self.load_metrics(job_ids, progress_bar, ignore_errors),
            cache_mode,
        )





class TrainingLogsViewer:
    @staticmethod
    def get_metric_by_job(df, metric, dock_to_bottom = False) -> pd.DataFrame:
        rdf = df.loc[df.metric == metric]
        rdf = rdf.feed(fluq.add_ordering_column('job_id', ('ordinal', not dock_to_bottom), 'order'))
        rdf = rdf.pivot(index='order', columns='job_id', values='value')
        rdf = rdf.sort_index(ascending=not dock_to_bottom)
        rdf.index=list(range(rdf.shape[0]))
        return rdf

    @staticmethod
    def get_last_values(df) -> pd.DataFrame:
        return (df
                .feed(fluq.add_ordering_column(['job_id', 'metric'], ('ordinal', False)))
                .feed(lambda z: z.loc[z.order == 0])
                .pivot_table(columns='metric', values='value', index='job_id')
        )

    @staticmethod
    def draw_overfit(df, x, y, ax=None, rolling_mean = None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 7))
        xdf = TrainingLogsViewer.get_metric_by_job(df, x)
        ydf = TrainingLogsViewer.get_metric_by_job(df, y)
        if rolling_mean is not None:
            xdf = xdf.rolling(rolling_mean).mean()
            ydf = ydf.rolling(rolling_mean).mean()
        for c in xdf.columns:
            ax.plot(xdf[c], ydf[c], label=c)
        mn = max(xdf.min().min(), ydf.min().min())
        mx = min(xdf.max().max(), ydf.max().max())
        ax.plot([mn, mx], [mn, mx], '--', c='red')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()

    @staticmethod
    def get_time_spans_between_iterations(df) -> pd.DataFrame:
        qdf = df
        qdf = qdf.loc[qdf.metric == 'iteration'].copy()
        qdf['next_ordinal'] = qdf.ordinal + 1
        qdf = qdf.merge(
            qdf.set_index(['job_id', 'ordinal']).rename(columns={'timestamp': 'next_timestamp'}).next_timestamp,
            left_on=['job_id', 'next_ordinal'],
            right_index=True)
        qdf['delta'] = (qdf.next_timestamp - qdf.timestamp).dt.total_seconds()
        return qdf

    @staticmethod
    def get_name_part_as_column(series, prefix, tp=None):
        regex_str = f'-{prefix}([^-]*)-'
        s = series.str.extract(regex_str)
        if tp is not None:
            s = s.astype(tp)
        return s