from typing import *
from ...._common import Loc, FileSyncer, Logger
from ...access import DataSource
from ..simple.featurization_job import FeaturizationJob, StreamFeaturizer
from .updatable_dataset import UpdatableDataset
import botocore

from datetime import datetime
from yo_fluq_ds import *
from uuid import uuid4
from pathlib import Path




class UpdatableFeaturizationJob:
    def __init__(self,
                 name: str,
                 version: str,
                 full_data_source: DataSource,
                 update_data_source_factory: Optional[Callable[[datetime,datetime],DataSource]],
                 featurizers: Dict[str, StreamFeaturizer],
                 syncer: FileSyncer,
                 location: Optional[Union[Path, str]] = None,
                 limit: Optional[int] = None,
                 reporting_frequency: Optional[int] = None,
                 force_major = False
                 ):
        self.name = name
        self.version = version
        self.full_data_source = full_data_source
        self.update_data_source_factory = update_data_source_factory
        self.featurizers = featurizers
        self.force_major = force_major


        if location is None:
            self.location = Loc.temp_path/'updatable_featurization_job'/str(uuid4())
        else:
            self.location = Path(location)

        self.syncer = syncer.change_local_folder(self.location)


        self.limit = limit
        self.reporting_frequency = reporting_frequency

        os.makedirs(self.location, exist_ok=True)

        if self.reporting_frequency is None and self.limit is not None:
            self.reporting_frequency = int(self.limit/10)

    def get_name_and_version(self) -> Tuple[str,str]:
        return self.name, self.version


    def should_be_major(self, force_major, last_record: UpdatableDataset.DescriptionItem):
        if self.force_major:
            return True
        if force_major:
            return True
        if self.update_data_source_factory is None:
            return True
        if last_record is None:
            return True
        if last_record.version!=self.version:
            return True
        return False



    def get_inner_job_and_record(self,
                                 force_major: bool,
                                 partition_uuid: str,
                                 records: List[UpdatableDataset.DescriptionItem],
                                 current_date: datetime) -> Tuple[FeaturizationJob, UpdatableDataset.DescriptionItem]:
        if len(records)==0:
            last_record = None
        else:
            last_record = records[-1]

        is_major = self.should_be_major(force_major, last_record)
        if is_major:
            src = self.full_data_source
        else:
            src = self.update_data_source_factory(last_record.timestamp, current_date)

        job = FeaturizationJob(
            self.name,
            self.version,
            src,
            self.featurizers,
            self.syncer.cd(partition_uuid),
            self.location/partition_uuid,
            self.reporting_frequency,
            self.limit
        )

        record = UpdatableDataset.DescriptionItem(
            partition_uuid,
            current_date,
            is_major,
            self.version
        )

        return job, record

    def run(self, current_time = None, force_major_update = None, custom_revision_id = None):
        if current_time is None:
            current_time = datetime.now()

        if force_major_update is None:
            force_major_update = False



        Logger.info(f'Starting lesvik job {self.name}, version {self.version}')
        Logger.info(f'Additional settings limit {self.limit or "NONE"}, reporting {self.reporting_frequency or "NONE"}')

        records_path = self.location/UpdatableDataset.DescriptionHandler.get_description_filename()
        if self.syncer.download_file(UpdatableDataset.DescriptionHandler.get_description_filename()) is not None:
            records = UpdatableDataset.DescriptionHandler.read_parquet(records_path)
        else:
            records = []

        Logger.info(f"{len(records)} previous revisions are found")
        if custom_revision_id is None:
            uuid = str(uuid4())
        else:
            uuid = custom_revision_id
        job, new_record = self.get_inner_job_and_record(force_major_update,uuid, records, current_time)
        Logger.info(f"Running with id {new_record.name} at {new_record.timestamp}, revision is {'MAJOR' if new_record.is_major else 'MINOR'}")
        job.run()

        if job.records_processed_ == 0:
            Logger.info('No records were processed. The update is not applied')
        else:
            Logger.info(f"{job.records_processed_} were processed")
            Logger.info("Uploading new description")
            updated_records = records+[new_record]
            UpdatableDataset.DescriptionHandler.write_parquet(updated_records, records_path)
            self.syncer.upload_file(UpdatableDataset.DescriptionHandler.get_description_filename())

        Logger.info("Job finished")

