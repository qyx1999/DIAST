import numpy as np
import tempfile
import shutil
import os
from PIL import Image
import subprocess
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def predict(
        self,
        image: Path = Input(
            description="Input Image.",
        ),
    ) -> Path:
        input_dir = "input_dir"
        output_path = Path(tempfile.mkdtemp()) / "output.png"

        try:
            for d in [input_dir, "results"]:
                if os.path.exists(input_dir):
                    shutil.rmtree(input_dir)
            os.makedirs(input_dir, exist_ok=False)

            input_path = os.path.join(input_dir, os.path.basename(image))
            shutil.copy(str(image), input_path)
            subprocess.call(
                [
                    "python",
                    "net/test.py",
                    "-opt",
                    "options/test/SRx4.yml",
                ]
            )
            res_dir = os.path.join(
                "results", "SRx4", "visualization", "custom"
            )
            assert (
                len(os.listdir(res_dir)) == 1
            ), "Should contain only one result for Single prediction."
            res = Image.open(os.path.join(res_dir, os.listdir(res_dir)[0]))
            res.save(str(output_path))

        finally:
            pass
            shutil.rmtree(input_dir)
            shutil.rmtree("results")

        return output_path
