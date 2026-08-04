from __future__ import annotations

import unittest

import pandas as pd

from factors.registry import (
    FACTOR_REGISTRY,
    build_factor,
    compute_factor,
    get_factor_metadata,
    register_factor,
)


class RegistryTests(unittest.TestCase):
    factor_name = "__test_factor__"

    def setUp(self) -> None:
        FACTOR_REGISTRY.pop(self.factor_name, None)

        @register_factor(name=self.factor_name, default_params={"scale": 2})
        def factory(params: dict[str, object] | None):
            assert params is not None
            scale = int(params["scale"])
            params["scale"] = 99
            return lambda frame: frame["close"] * scale

    def tearDown(self) -> None:
        FACTOR_REGISTRY.pop(self.factor_name, None)

    def test_default_params_are_isolated(self) -> None:
        build_factor(self.factor_name)
        metadata = get_factor_metadata(self.factor_name)
        self.assertEqual(metadata["default_params"], {"scale": 2})
        metadata["default_params"]["scale"] = 50
        self.assertEqual(
            get_factor_metadata(self.factor_name)["default_params"],
            {"scale": 2},
        )

    def test_compute_factor_checks_result_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [10.0],
            }
        )
        result = compute_factor(self.factor_name, frame)
        pd.testing.assert_series_equal(result, frame["close"] * 2)


if __name__ == "__main__":
    unittest.main()
