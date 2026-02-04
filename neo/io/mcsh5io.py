from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.mcsh5rawio import McsH5RawIO


class McsH5IO(McsH5RawIO, BaseFromRaw):
    def __init__(self, filename):
        McsH5RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
