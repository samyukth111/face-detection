import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_without_errors(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"

        app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Face Finder")
        self.assertEqual(len(app.file_uploader), 1)


if __name__ == "__main__":
    unittest.main()
