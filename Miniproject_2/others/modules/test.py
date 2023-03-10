import re
import sys
import unittest
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# Check python version
version = sys.version_info[:2]
if version < (3, 6):
    raise RuntimeError("This script uses f-strings, which requires Python version >= 3.6. Use a newer version of Python.")


"""
Note to students:

1. If you want to import files from the "others" folder
(e.g. from "others/nn.py"), you should write the import statement as:
   from .others.nn import Module

2. To load your saved model, you can use the following code:
   from pathlib import Path
   model_path = Path(__file__).parent / "bestmodel.pth"
   model = torch.load(model_path)

3. Run this script with the command:
   python3 test.py -d path_to_data_folder -p path_to_root_project_folder

4. More tests will be present in the final test.py that we will be running
"""

class Tests(unittest.TestCase):
    @staticmethod
    def compute_psnr(x, y, max_range=1.0):
        assert x.shape == y.shape and x.ndim == 4
        return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()


    def test_folder_structure(self):
        title("Testing folder structure")
        self.assertTrue(project_path.exists(), f"No folder found at {project_path}")
        self._test_folder_structure(1)
        self._test_folder_structure(2)

    def _test_folder_structure(self, project_number):
        miniproject_path = project_path / f"Miniproject_{project_number}"
        self.assertTrue(miniproject_path.exists(), f"No folder Miniproject_{project_number} found at {project_path}")
        
        for file in ["__init__.py", "model.py"]:
            with self.subTest(f"Checking file {file} for project {project_number}"):
                self.assertTrue((miniproject_path / file).exists(), f"No file {file} found at {miniproject_path}")
        
        for file in [f"Report_{project_number}.pdf", "bestmodel.pth"]:
            if not (miniproject_path / file).exists():
                warn(f"Miniproject folder {project_number} does not contain a {file} file")

    def test_instantiate_model_class(self):
        title("Testing model class instantiation")
        for i in [1,2]:
            with self.subTest(f"Checking instantiate model class for project {i}"):
                self._test_instantiate_model_class(i)

    def _test_instantiate_model_class(self, project_number):
        model = importlib.import_module(f"Miniproject_{project_number}.model")
        model.Model()


    def test_forward_dummy_input(self):
        title("Testing forward dummy input")
        for i in [1,2]:
            with self.subTest(f"Checking forward dummy input for project {i}"):
                self._test_forward_dummy_input(i)

    def _test_forward_dummy_input(self, project_number):
        Model = importlib.import_module(f"Miniproject_{project_number}.model").Model
        model = Model()
        out = model.predict(torch.rand(1, 3, 512, 512) * 255)
        self.assertEqual(out.shape, (1, 3, 512, 512))
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 255)

        if out.max() <= 1:
            warn("The output of the predict function should be a Tensor in the range [0, 255]")


    def test_model_pnsr(self):
        title("Testing pretrained model")
        for i in [1,2]:
            with self.subTest(f"Testing pretrained model for project {i}"):
                self._test_model_pnsr(i)

    def _test_model_pnsr(self, project_number):
        Model = importlib.import_module(f"Miniproject_{project_number}.model").Model
        model = Model()
        model.load_pretrained_model()

        val_path = data_path / "val_data.pkl"
        val_input, val_target = torch.load(val_path)
        val_target = val_target.float() / 255.0

        mini_batch_size = 100
        model_outputs = []
        for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
            output = model.predict(val_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output.cpu())
        model_outputs = torch.cat(model_outputs, dim=0) / 255.0

        output_psnr = self.compute_psnr(model_outputs, val_target)
        print(f"[PSNR {project_number}: {output_psnr:.2f} dB]")


    def test_train_model(self):
        title("Testing model training")
        for i in [1,2]:
            with self.subTest(f"Testing model training for project {i}"):
                self._test_train_model(i)

    def _test_train_model(self, project_number):
        Model = importlib.import_module(f"Miniproject_{project_number}.model").Model
        model = Model()
        model.load_pretrained_model()

        train_path = data_path / "train_data.pkl"
        val_path = data_path / "val_data.pkl"
        train_input0, train_input1 = torch.load(train_path)
        val_input, val_target = torch.load(val_path)
        val_target = val_target.float() / 255.0

        output_psnr_before = self.compute_psnr(val_input, val_target)

        model.train(train_input0, train_input1, num_epochs=1)

        mini_batch_size = 100
        model_outputs = []
        for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
            output = model.predict(val_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output.cpu())
        model_outputs = torch.cat(model_outputs, dim=0) / 255.0

        output_psnr_after = self.compute_psnr(model_outputs, val_target)
        print(f"[PSNR {project_number}: {output_psnr_after:.2f} dB]")
        self.assertGreater(output_psnr_after, output_psnr_before)


    def test_framework_block(self):
        title("Testing blocks")
        model_module = importlib.import_module(f"Miniproject_2.model")

        x = torch.randn(1, 3, 32, 32)

        with self.subTest("Testing convolution"):
            Conv2d = model_module.Conv2d
            conv = Conv2d(3, 3, 3)
            self.assertTrue(torch.allclose(conv.forward(x), F.conv2d(x, conv.weight, conv.bias)))

        with self.subTest("Testing sigmoid"):
            Sigmoid = model_module.Sigmoid
            sigmoid = Sigmoid()
            self.assertTrue(torch.allclose(sigmoid.forward(x), torch.sigmoid(x)))

        with self.subTest("Testing sequential"):
            Sequential = model_module.Sequential
            seq = Sequential(conv, sigmoid)
            self.assertTrue(torch.allclose(seq.forward(x), F.conv2d(x, conv.weight, conv.bias).sigmoid()))

def warn(msg):
    print(f"\33[33m!!! Warning: {msg}\33[39m")

def title(msg):
    print(f"\n=============\n> {msg} ...")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--project-path', help='Path to the project folder', required=True)
    parser.add_argument('-d', '--data-path', help='Path to the data folder', required=True)
    args = parser.parse_args()
    
    project_path = Path(args.project_path)
    data_path = Path(args.data_path)

    if re.match(r'^Proj(_(\d{6})){3}$', project_path.name) is None:
        warn("Project folder name must be in the form Proj_XXXXXX_XXXXXX_XXXXXX")

    sys.path.append(args.project_path)
    unittest.main(argv=[''], verbosity=0)
