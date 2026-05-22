"""Tiny launcher because bundled openvino.tools.benchmark.main lacks __main__ guard."""
import sys
from openvino.tools.benchmark.main import main
sys.exit(main() or 0)
