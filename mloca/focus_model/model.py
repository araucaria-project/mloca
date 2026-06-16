import pickle
from importlib import resources
from pathlib import Path
from typing import Optional
import numpy as np


class Model:

    def __init__(self, telescope_id: str, models_dir: Optional[str | Path] = None):
        self.model = None
        self.telescope_id = telescope_id
        self.models_dir = Path(models_dir) if models_dir is not None else None
        super().__init__()

    @property
    def model_filename(self) -> str:
        return f'{self.telescope_id}_random_forest_focus_model.pkl'

    async def load_model(self):
        try:
            if self.models_dir is not None:
                model_path = self.models_dir / self.model_filename
            else:
                model_path = resources.files(__package__).joinpath('models', self.model_filename)

            with model_path.open('rb') as f:
                self.model = pickle.load(f)
        except OSError:
            pass

    async def save_model(self):
        try:
            models_dir = self.models_dir or Path(__file__).resolve().parent / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)

            with (models_dir / self.model_filename).open('wb') as f:
                pickle.dump(self.model, f)
        except OSError:
            pass

    async def predict(self, temp: float, hum: float) -> Optional[float]:
        try:
            ret = float(self.model.predict(np.array([[temp, hum]]))[0])
            return ret
        except ValueError:
            return None
