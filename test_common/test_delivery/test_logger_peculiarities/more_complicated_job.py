from ....delivery.delivery import *
from ...._common import Logger
from .job_in_package import JobInPackage

class JobInstallingPackage:
    def run(self):
        packaging = Packaging(
            'test_job_internal',
            '2',
            dict(job=JobInPackage())
        )
        packaging.silent = True
        pkg_info = packaging.make_package()
        loader = install_package_and_get_loader(pkg_info.package_location, silent = True)

        job = loader.load_resource('job')
        result = job.run()
        if result!='ok':
            Logger.info('Error')
        else:
            Logger.info('Success')
        Logger.info(type(Logger))


