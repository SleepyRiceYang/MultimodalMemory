import json
import unittest
import numpy as np
from dense_retrieval import cosine_top_k, parse_decision


class DenseTests(unittest.TestCase):
    def test_cosine_sort_has_no_threshold_and_is_deterministic(self):
        entries = [{'skill': s, 'name': s, 'description': s} for s in ('c', 'a', 'b')]
        vectors = np.asarray([[0, 1], [1, 0], [-1, 0]], dtype=np.float32)
        hits = cosine_top_k(entries, vectors, np.array([1, 0]), 3)
        self.assertEqual([c['skill'] for c in hits], ['a', 'c', 'b'])
        self.assertEqual(hits[-1]['cosine_similarity'], -1)

    def test_accepts_zero_to_three_but_not_duplicates_or_unknowns(self):
        candidates = [{'skill': s} for s in ('a', 'b', 'c', 'd')]
        for value in (None, [], ['a'], ['c', 'b', 'a']):
            names, _ = parse_decision(json.dumps({'skills': value, 'reason': 'test'}), candidates)
            self.assertEqual(names, value or [])
        for value in (['a', 'b', 'c', 'd'], ['a', 'a'], ['unknown'], 'a', [{}]):
            with self.assertRaises(ValueError):
                parse_decision(json.dumps({'skills': value, 'reason': 'test'}), candidates)


if __name__ == '__main__':
    unittest.main()
