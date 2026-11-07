"""Regression tests for taxonomy helpers shared by data and evaluation."""

import contextlib
import io
import unittest

from llavaguard.taxonomy.PEGI.PEGI_Graph import (
    get_max_pegi_score,
    graph_with_ratings,
    policy_graph,
    remove_mutually_exclusive_categories,
)


class PegiGraphTests(unittest.TestCase):
    def test_numbered_severity_groups_keep_only_the_highest_label(self):
        groups = [
            ['2. Unrealistic and Implied Violence', '4. Mild Violence', '6. Realistic Violence', '9. Gross Violence'],
            ['11. Mild Swearing', '12. Offensive Language'],
            ['16. Sexual Innuendo', '17. Mild Nudity Content', '19. Explicit Nudity Content', '18. Explicit Sexual Content'],
            ['21. Scary Content', '22. Moderately Disturbing Content', '24. Highly Disturbing Content'],
            ['27. Non-realistic Self-harm', '29. Self-injury', '28. Suicide'],
        ]
        for labels in groups:
            with self.subTest(labels=labels):
                self.assertEqual(remove_mutually_exclusive_categories(labels), [labels[-1]])

    def test_cleanup_keeps_unrelated_labels_and_does_not_mutate_input(self):
        labels = ['1. Educational Content', '4. Mild Violence', '9. Gross Violence', '40. Alcohol']
        original = list(labels)
        self.assertEqual(remove_mutually_exclusive_categories(labels),
                         ['1. Educational Content', '9. Gross Violence', '40. Alcohol'])
        self.assertEqual(labels, original)

    def test_safe_and_mixed_ratings(self):
        for labels in ([], ['NA: None applying'], 'NA: None applying'):
            self.assertEqual(get_max_pegi_score(labels), 3)
        self.assertEqual(get_max_pegi_score(['3. Military and Warfare', '9. Gross Violence']), 18)

    def test_unknown_or_unparsed_labels_fail_clearly(self):
        for labels in (['Unknown label'], '3. Military and Warfare'):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                get_max_pegi_score(labels)

    def test_illegal_labels_are_not_rated_safe(self):
        with self.assertRaises(ValueError):
            get_max_pegi_score(['46. Child Endangerment'])

    def test_custom_graph_is_used(self):
        graph = {'Custom category': {'Custom label': {'rating': 'PEGI 16'}}}
        self.assertEqual(get_max_pegi_score(['Custom label'], graph), 16)

    def test_printer_includes_all_categories_and_subcategories(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            text = graph_with_ratings()
        self.assertEqual(output.getvalue().rstrip(), text)
        for category, subcategories in policy_graph.items():
            self.assertIn(category, text)
            for subcategory in subcategories:
                self.assertIn(f'"{subcategory}"', text)


if __name__ == '__main__':
    unittest.main()
