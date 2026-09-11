"""RPC flags must survive Python-to-Lua serialization as actual booleans."""
import unittest

from lupa import LuaRuntime

from .export import lua_literal


class LuaLiteralTests(unittest.TestCase):
    def test_nested_boolean_flags_survive_lua_evaluation(self):
        lua = LuaRuntime()
        result = lua.execute('return ' + lua_literal({
            'enabled': True, 'disabled': False, 'values': [True, False, 1, 0],
        }))
        self.assertIs(result['enabled'], True)
        self.assertIs(result['disabled'], False)
        self.assertIs(result['values'][1], True)
        self.assertIs(result['values'][2], False)
        self.assertEqual(lua.eval('type')(result['values'][3]), 'number')
        self.assertEqual(result['values'][4], 0)

    def test_nonfinite_values_remain_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                lua_literal(value)


if __name__ == '__main__':
    unittest.main()
