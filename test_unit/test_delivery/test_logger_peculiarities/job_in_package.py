from ...._common import Logger

class JobInPackage:
    def run(self):
        Logger.info('I am in package')
        Logger.info(type(Logger))
        return 'ok'
