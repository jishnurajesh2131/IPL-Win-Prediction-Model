"""
Unit tests for venue standardization and city imputation.
"""

import unittest
from src.data_processing.venue_mapping import (
    map_venue_name,
    impute_city_if_missing,
    CANONICAL_VENUE_MAP,
)


class TestVenueMapping(unittest.TestCase):
    def test_chinnaswamy_aliases(self):
        aliases = [
            "M Chinnaswamy Stadium",
            "M Chinnaswamy Stadium, Bengaluru",
            "M.Chinnaswamy Stadium",
        ]
        for v in aliases:
            self.assertEqual(map_venue_name(v), "M Chinnaswamy Stadium")

    def test_eden_gardens_aliases(self):
        aliases = ["Eden Gardens", "Eden Gardens, Kolkata"]
        for v in aliases:
            self.assertEqual(map_venue_name(v), "Eden Gardens")

    def test_wankhede_aliases(self):
        aliases = ["Wankhede Stadium", "Wankhede Stadium, Mumbai"]
        for v in aliases:
            self.assertEqual(map_venue_name(v), "Wankhede Stadium")

    def test_renamed_stadiums(self):
        self.assertEqual(map_venue_name("Feroz Shah Kotla"), "Arun Jaitley Stadium")
        self.assertEqual(map_venue_name("Arun Jaitley Stadium, Delhi"), "Arun Jaitley Stadium")
        self.assertEqual(map_venue_name("Sardar Patel Stadium, Motera"), "Narendra Modi Stadium")
        self.assertEqual(map_venue_name("Subrata Roy Sahara Stadium"), "Maharashtra Cricket Association Stadium")

    def test_distinct_mumbai_grounds_not_merged(self):
        """Ensure Wankhede, Brabourne, and DY Patil are not merged."""
        v_wankhede = map_venue_name("Wankhede Stadium")
        v_brabourne = map_venue_name("Brabourne Stadium")
        v_dypatil = map_venue_name("Dr DY Patil Sports Academy")

        self.assertNotEqual(v_wankhede, v_brabourne)
        self.assertNotEqual(v_wankhede, v_dypatil)
        self.assertNotEqual(v_brabourne, v_dypatil)

    def test_city_imputation(self):
        # Unambiguous imputation for Dubai and Sharjah
        self.assertEqual(impute_city_if_missing("Dubai International Cricket Stadium", ""), "Dubai")
        self.assertEqual(impute_city_if_missing("Dubai International Cricket Stadium", None), "Dubai")
        self.assertEqual(impute_city_if_missing("Sharjah Cricket Stadium", ""), "Sharjah")

        # Existing city preserved
        self.assertEqual(impute_city_if_missing("Eden Gardens", "Kolkata"), "Kolkata")
        self.assertEqual(impute_city_if_missing("Wankhede Stadium", "Mumbai"), "Mumbai")

        # Do not guess ambiguous cities
        self.assertEqual(impute_city_if_missing("Some Unknown Ground", ""), "")


if __name__ == "__main__":
    unittest.main()
