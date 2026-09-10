import os
import unittest

import main


class TestServerConfig(unittest.TestCase):
    def test_get_server_port_defaults_to_8001(self):
        self.assertEqual(main.get_server_port(), 8001)

    def test_get_server_port_reads_env(self):
        original = os.environ.get("PORT")
        os.environ["PORT"] = "8123"
        try:
            self.assertEqual(main.get_server_port(), 8123)
        finally:
            if original is None:
                os.environ.pop("PORT", None)
            else:
                os.environ["PORT"] = original


if __name__ == "__main__":
    unittest.main()
