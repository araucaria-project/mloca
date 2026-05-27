import pickle
from typing import Optional
import numpy as np


class Model:

    def __init__(self, telescope_id: str):
        self.model = None
        self.telescope_id = telescope_id
        super().__init__()

    async def load_model(self):
        try:
            with open(f'models/{self.telescope_id}_random_forest_focus_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except OSError:
            pass

    async def save_model(self):
        try:
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        except OSError:
            pass

    async def predict(self, temp: float, hum: float) -> Optional[float]:
        try:
            ret = float(self.model.predict(np.array([[temp, hum]]))[0])
            return ret
        except ValueError:
            return None
