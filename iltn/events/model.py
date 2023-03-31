import tensorflow as tf

import iltn

class TrapezoidalModel:
    ap_index: int = 0
    bp_index: int = 1
    cp_index: int = 2
    dp_index: int = 3


    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    def a(self, inputs:list[iltn.terms.Constant]):
        pass
    
    def b(self, inputs:list[iltn.terms.Constant]):
        pass

    def c(self, inputs:list[iltn.terms.Constant]):
        pass

    def d(self, inputs:list[iltn.terms.Constant]):
        pass

    def embedding(self, inputs: list[iltn.terms.Constant]):
        pass

