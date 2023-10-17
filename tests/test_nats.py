"""
Testing script for the NATS Interface. This test is specific to lookup enabled instances of NATS Interface.
"""

import sys
import os

# Get the path to parentfolder
parentfolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parentfolder_path)

from src import NATS_Interface
from src.network_utils import TinyNetwork
from src.utils import get_project_root
import unittest
import numpy as np

class TestNATSInterface_Lookup(unittest.TestCase):

    def setUp(self):
        self.nats_interface = NATS_Interface(
            use_lookup_table=True,
            path_to_lookup=str(get_project_root()) + "/searchspaces/nats_interface_lookuptable.json",
            path_to_lookup_index=str(get_project_root()) + "/searchspaces/nats_arch_index.json",
        )

    def test_init(self):
        self.assertEqual(self.nats_interface._dataset, "cifar10")
        self.assertEqual(self.nats_interface.target_device, "edgegpu")
        """Only tests the init method of the NATS_Interface class when lookup is enabled."""
        self.assertTrue(self.nats_interface.using_lookup)
        
    def test_dataset_property(self):
        self.assertEqual(self.nats_interface.dataset, "cifar10")

    def test_getitem(self):
        # You'll need to figure out what the expected return value should be
        expected_output = "{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*["none" for _ in range(6)])
        self.assertEqual(self.nats_interface.__getitem__(0), expected_output)

    def test_get_config_dictionary(self):
        # Similarly, determine the expected output
        expected_output = dict(
            name="infer.tiny", 
            C=16,
            N=5, 
            num_classes=self.nats_interface.network_numclasses,
            arch_str=self.nats_interface[10]
        )
        self.assertEqual(self.nats_interface.get_config_dictionary(10), expected_output)

    def test_get_network(self):
        # You might want to check the type of the returned object and/or some of its attributes
        network = self.nats_interface.get_network(0)
        self.assertIsInstance(network, TinyNetwork)
        
    def test_compute_score(self):
        # Mocking or setting up any required state before calling the method
        score_values = {
            "naswot": 1e-6,
            "logsynflow": 57887.68851769018,
            "skip": 0.0
        }
        for score in ["naswot", "logsynflow", "skip"]:
            score_value = self.nats_interface.compute_score(f"{score}_score", 0)
            self.assertIsInstance(score_value, float)
            self.assertEqual(score_value, score_values[score])

    def test_get_score_mean_and_std(self):
        mean, std = self.nats_interface.get_score_mean_and_std("naswot_score", sample_size=100)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)

    def test_generate_random_samples(self):
        cell_structures, idxs = self.nats_interface.generate_random_samples()
        self.assertIsInstance(cell_structures, list)
        self.assertIsInstance(idxs, np.ndarray)
        self.assertEqual(len(cell_structures), 10)
    
if __name__ == '__main__':
    unittest.main()

