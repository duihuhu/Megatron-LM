class DummyWriteItem:
    def __init__(self, item_type=None, index: int = 0, name: str = None):
        self.type = item_type
        self.index = index
        self.name = name
        self.storage_key = None
        self.offset = None
        self.length = None