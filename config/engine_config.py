from dataclasses import dataclass

@dataclass
class BasicEngineConfig:
    # data parameters
    data_path: str | None = "/home/lab/lmc/dirty_hands/data/mobvoi_seq_monkey_general_open_corpus.jsonl"
    
    # training parameters
    epoch: int = 100
    lr: float = 3e-4
    batch: int = 16
    
    # log parameters
    logger: str | None = None
    
    # runtime parameters
    seed: int = 0
    device: str = 'cuda'
    

@dataclass
class PretrainConfig(BasicEngineConfig):
    pass