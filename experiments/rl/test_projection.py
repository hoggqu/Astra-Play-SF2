"""Offline projected-teacher checks; no emulator, ROM, or model training."""
import unittest
from .projection_teacher import ProjectionTeacher, project_sequence


def state(opponent=2, x=500, y=40, action=0):
    return {'p1': {'char': 4, 'hp': 144, 'x': 200, 'y': 40, 'a': 0, 'anim': 1},
            'p2': {'char': opponent, 'hp': 144, 'x': x, 'y': y, 'a': action, 'anim': 1}, 'timer': 99}


class ProjectionTests(unittest.TestCase):
    def test_directional_sequences_use_complete_prefix_and_both_sides(self):
        for forward in ('L', 'R'):
            self.assertEqual(project_sequence([(2, forward), (2, 'D'), (2, 'D '+forward+' MP'), (12, '')], forward)[0], 13)
            self.assertEqual(project_sequence([(3, 'D'), (3, 'D '+forward), (3, forward+' LP'), (16, '')], forward)[0], 12)
            back = 'L' if forward == 'R' else 'R'
            self.assertEqual(project_sequence([(2, 'D '+back)], forward)[0], 3)
            self.assertEqual(project_sequence([(3, forward+' HP'), (4, back)], forward)[0], 7)

    def test_normal_and_jump_projection_does_not_call_guard_a_fireball(self):
        cases = [([(2, 'D L')], 3), ([(4, 'D HK'), (12, 'D L')], 9),
                 ([(20, 'U R'), (4, 'HK')], 14), ([(6, 'U R')], 4),
                 ([(6, 'U L')], 5), ([(3, 'HK'), (3, 'L')], 11), ([(2, '')], 0)]
        for sequence, expected in cases:
            self.assertEqual(project_sequence(sequence, 'R')[0], expected)

    def test_all_frozen_opponent_selectors_run_without_mame(self):
        teacher = ProjectionTeacher()
        self.assertTrue(teacher.identity['lossy'])
        for opponent in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11):
            for distance in (20, 80, 200, 400):
                teacher.reset()
                for y, action in ((40, 0), (100, 10), (80, 12), (40, 14)):
                    move, info = teacher.predict(state(opponent, 200+distance, y, action))
                    self.assertIn(move, range(15))
                    self.assertTrue(info['sequence'])

    def test_sampled_velocity_age_rebound_and_reset(self):
        teacher = ProjectionTeacher()
        teacher.predict(state(y=100))
        _, info = teacher.predict(state(y=76, action=10))
        self.assertEqual(info['estimated_vy'], -2)
        self.assertEqual(info['estimated_attack_age'], 0)
        _, info = teacher.predict(state(y=88, action=10))
        self.assertTrue(info['estimated_rebound'])
        self.assertEqual(info['estimated_attack_age'], 12)
        teacher.reset()
        _, info = teacher.predict(state(y=40))
        self.assertEqual(info['estimated_vy'], 0)
        self.assertFalse(info['estimated_rebound'])
        self.assertEqual(info['estimated_attack_age'], 999)


if __name__ == '__main__':
    unittest.main()
