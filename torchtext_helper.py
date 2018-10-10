import collections

class HelperIterator:
    def __init__(self, *, iterator, fields):
        self.iterator = iterator
        self.field_names = [x[0] for x in fields]
        
    def __iter__(self):
        NamedTup = collections.namedtuple('HelperBatch', self.field_names)
        for batch in self.iterator:
            values = [getattr(batch, f) for f in self.field_names]
            yield NamedTup(*values)
    
    def __len__(self):
        return len(self.iterator)