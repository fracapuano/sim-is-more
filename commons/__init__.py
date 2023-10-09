from .tables import *
from .interfaces import *
from .genetics import *
from .utils import *

SearchSpaceDict = {
    "nats": lambda dataset: NATSInterface(dataset=dataset),
    "fast_nats": lambda dataset: NATS_FastInterface(dataset=dataset)
}