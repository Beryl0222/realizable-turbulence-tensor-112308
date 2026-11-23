"""核对交换资料的字段和数值关系。"""
import json
from pathlib import Path
import unittest

import numpy as np


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "data" / "exchange.json"
        self.data = json.loads(source.read_text(encoding="utf-8"))

    def test_exchange_data(self):
        ridge,bay,zero=self.data['tiles']
        self.assertLess(np.linalg.eigvalsh(ridge['base'])[0],0)
        self.assertGreater(sum(item[2] for item in bay['hard']),bay['trace'])
        self.assertEqual(zero['trace'],0)
        np.testing.assert_allclose(np.array(ridge['frame']).T @ ridge['frame'],np.eye(3))
