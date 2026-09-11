"""Pure shared R1 predicates. No I/O, emulator imports, or game commands."""
def make_opening_guard(difficulty_bits):
    assert type(difficulty_bits) is int and 0 <= difficulty_bits <= 4

    def opening_context(s, opponent):
        return (isinstance(s, dict) and s.get('paused') is True and s.get('controller_busy') is False
                and s.get('difficulty_bits') == difficulty_bits
                and isinstance(s.get('p1'), dict) and isinstance(s.get('p2'), dict)
                and s['p1'].get('character') == 4 and s['p2'].get('character') == opponent)

    def visible_opening(s, opponent):
        # Unchanged Core-compatible live-R1 rule: no added neutral frames.
        return (opening_context(s, opponent) and s.get('timer_seconds') == 99
                and all(p.get('hp') == 144 and p.get('displayed_hp') == 144 and p.get('round_wins') == 0
                        and p.get('y') == 40 and type(p.get('x')) is int and p['x'] > 0
                        and type(p.get('animation')) is int and p['animation'] > 0
                        for p in (s['p1'], s['p2'])))

    def uninitialized_intro(s, opponent):
        if not opening_context(s, opponent):
            return False
        players = (s['p1'], s['p2'])
        # Existing all-zero actors / paired HP rule, including stale intro timer.
        old = all(all(p.get(k) == 0 for k in ('x','y','action','animation','round_wins','timeout_hp'))
                  and (p.get('hp'), p.get('displayed_hp')) in ((0,0),(144,144)) for p in players)
        # Coin20-l6-04 / 16163: both ground positions set, neither animation loaded.
        positioned = (s.get('timer_seconds') == 99
                      and all(p.get('hp') == 144 and p.get('displayed_hp') == 144
                              and type(p.get('x')) is int and p['x'] > 0 and p.get('y') == 40
                              and all(p.get(k) == 0 for k in ('action','animation','round_wins','timeout_hp'))
                              for p in players))
        return old or positioned

    def await_visible_opening(s, opponent, advance_intro):
        for retry in range(5):
            if visible_opening(s, opponent):
                return s
            if not uninitialized_intro(s, opponent):
                raise ValueError('Not a visible R1 or verified uninitialized intro; do not advance combat')
            if retry == 4:
                raise TimeoutError('Introduction still uninitialized after four 600-frame waits; no further advance')
            s = advance_intro(s, opponent, 600)
        raise AssertionError('Unreachable opening wait')

    return opening_context, visible_opening, uninitialized_intro, await_visible_opening
