import warnings
import sre_parse
from itertools import product

# Suppress the DeprecationWarning for sre_parse
warnings.filterwarnings("ignore", category=DeprecationWarning)

def split_at_top_level_or(pattern):
    """
    Splits the pattern at the top-level '|' operators, respecting parentheses and escaped characters.
    """
    subpatterns = []
    current = ''
    stack = []
    escaped = False
    for c in pattern:
        if escaped:
            current += c
            escaped = False
        elif c == '\\':
            current += c
            escaped = True
        elif c == '(':
            stack.append(c)
            current += c
        elif c == ')':
            if stack:
                stack.pop()
            current += c
        elif c == '|' and not stack:
            subpatterns.append(current)
            current = ''
        else:
            current += c
    if current:
        subpatterns.append(current)
    return subpatterns

def escape_literal_char(c):
    if c in ['\\', '_', '%']:
        return '\\' + c
    elif c == "'":
        return "''"
    else:
        return c

class Pattern:
    def __init__(self, pattern_str, anchored_start=False, anchored_end=False):
        self.pattern_str = pattern_str
        self.anchored_start = anchored_start
        self.anchored_end = anchored_end
        self.exclusions = []

    def __hash__(self):
        return hash((self.pattern_str, self.anchored_start, self.anchored_end))

    def __eq__(self, other):
        return (self.pattern_str, self.anchored_start, self.anchored_end) == (other.pattern_str, other.anchored_start, other.anchored_end)

def _process_subpattern(parsed_pattern):
    patterns = [Pattern('')]  # List of Pattern objects
    fully_translated = True

    for token in parsed_pattern:
        op, arg = token
        if op == sre_parse.LITERAL:
            c = chr(arg)
            c = escape_literal_char(c)  # Escape literals here
            old_patterns = patterns.copy()
            patterns = []
            for pat in old_patterns:
                new_p = Pattern(pat.pattern_str + c, pat.anchored_start, pat.anchored_end)
                new_p.exclusions = pat.exclusions.copy()
                patterns.append(new_p)
        elif op == sre_parse.AT:
            if arg == sre_parse.AT_END:
                for pat in patterns:
                    pat.anchored_end = True
            elif arg == sre_parse.AT_BEGINNING or arg == sre_parse.AT_BEGINNING_STRING:
                for pat in patterns:
                    pat.anchored_start = True
            else:
                return [], False
        elif op == sre_parse.ANY:
            old_patterns = patterns.copy()
            patterns = []
            for pat in old_patterns:
                new_p = Pattern(pat.pattern_str + '_', pat.anchored_start, pat.anchored_end)  # '_' is a wildcard, do not escape
                new_p.exclusions = pat.exclusions.copy()
                patterns.append(new_p)
        elif op == sre_parse.IN:
            chars = []
            negated = False
            for item in arg:
                if item[0] == sre_parse.NEGATE:
                    negated = True
                elif item[0] == sre_parse.LITERAL:
                    chars.append(chr(item[1]))
                elif item[0] == sre_parse.RANGE:
                    start_char = chr(item[1][0])
                    end_char = chr(item[1][1])
                    chars.extend([chr(c) for c in range(ord(start_char), ord(end_char) + 1)])
                else:
                    return [], False
            old_patterns = patterns.copy()
            if negated:
                new_patterns = []
                for pat in old_patterns:
                    p_with_placeholder = pat.pattern_str + '_'  # '_' is a wildcard, do not escape
                    position = len(pat.pattern_str)  # position of '_'
                    new_p = Pattern(p_with_placeholder, pat.anchored_start, pat.anchored_end)
                    new_p.exclusions = pat.exclusions.copy()
                    new_p.exclusions.append({'position': position, 'chars': chars.copy()})
                    new_patterns.append(new_p)
                patterns = new_patterns
            else:
                new_patterns = []
                for pat in old_patterns:
                    for c in chars:
                        c = escape_literal_char(c)  # Escape literals here
                        new_p = Pattern(pat.pattern_str + c, pat.anchored_start, pat.anchored_end)
                        new_p.exclusions = pat.exclusions.copy()
                        new_patterns.append(new_p)
                patterns = new_patterns
        elif op == sre_parse.BRANCH:
            _, options = arg
            all_branch_patterns = []
            for option in options:
                subpatterns, sub_fully_translated = _process_subpattern(option)
                if not sub_fully_translated:
                    return [], False
                for pat in patterns:
                    for subpat in subpatterns:
                        new_p = Pattern(pat.pattern_str + subpat.pattern_str, pat.anchored_start or subpat.anchored_start, pat.anchored_end or subpat.anchored_end)
                        # Adjust exclusions positions
                        new_p.exclusions = pat.exclusions.copy()
                        for exclusion in subpat.exclusions:
                            adjusted_exclusion = {
                                'position': exclusion['position'] + len(pat.pattern_str),
                                'chars': exclusion['chars']
                            }
                            new_p.exclusions.append(adjusted_exclusion)
                        all_branch_patterns.append(new_p)
            patterns = all_branch_patterns
        elif op == sre_parse.SUBPATTERN:
            subpattern = arg[3]
            subpatterns, sub_fully_translated = _process_subpattern(subpattern)
            if not sub_fully_translated:
                return [], False
            new_patterns = []
            for pat in patterns:
                for subpat in subpatterns:
                    new_p = Pattern(pat.pattern_str + subpat.pattern_str, pat.anchored_start or subpat.anchored_start, pat.anchored_end or subpat.anchored_end)
                    # Adjust exclusions positions
                    new_p.exclusions = pat.exclusions.copy()
                    for exclusion in subpat.exclusions:
                        adjusted_exclusion = {
                            'position': exclusion['position'] + len(pat.pattern_str),
                            'chars': exclusion['chars']
                        }
                        new_p.exclusions.append(adjusted_exclusion)
                    new_patterns.append(new_p)
            patterns = new_patterns
        elif op == sre_parse.MAX_REPEAT:
            min_repeat, max_repeat, subpattern = arg
            if min_repeat == 0 and max_repeat == sre_parse.MAXREPEAT:
                # Matches '.*', replace with '%'
                if len(subpattern) == 1 and subpattern[0][0] == sre_parse.ANY:
                    for pat in patterns:
                        pat.pattern_str += '%'  # '%' is a wildcard, do not escape
                else:
                    return [], False
            elif min_repeat == 0 and max_repeat == 1:
                # Handle '?', optional element
                subpatterns, sub_fully_translated = _process_subpattern(subpattern)
                if not sub_fully_translated:
                    return [], False
                # Patterns without the optional element
                patterns_without = patterns.copy()
                # Patterns with the optional element
                new_patterns = []
                for pat in patterns:
                    for subpat in subpatterns:
                        new_p = Pattern(pat.pattern_str + subpat.pattern_str, pat.anchored_start or subpat.anchored_start, pat.anchored_end or subpat.anchored_end)
                        # Adjust exclusions positions
                        new_p.exclusions = pat.exclusions.copy()
                        for exclusion in subpat.exclusions:
                            adjusted_exclusion = {
                                'position': exclusion['position'] + len(pat.pattern_str),
                                'chars': exclusion['chars']
                            }
                            new_p.exclusions.append(adjusted_exclusion)
                        new_patterns.append(new_p)
                patterns = patterns_without + new_patterns
            else:
                return [], False
        else:
            return [], False
    return patterns, fully_translated

def escape_like_pattern(s):
    # Only escape single quotes for SQL LIKE patterns
    s = s.replace("'", "''")
    return s

def regex_to_like(pattern):
    """
    Converts a regex pattern to an Oracle SQL LIKE query.
    """
    pis = split_at_top_level_or(pattern)
    like_clauses = []
    regexp_clauses = []
    any_translated = False
    all_translated = True

    for pi in pis:
        try:
            parsed = sre_parse.parse(pi)
        except Exception as e:
            print(f"Error parsing pattern '{pi}': {e}")
            all_translated = False
            regexp_clauses.append(f"REGEXP_LIKE(COL, '{pi}')")
            continue

        patterns, fully_translated = _process_subpattern(parsed)
        if fully_translated and patterns:
            any_translated = True
            # Build LIKE clauses
            positive_clauses = []
            negative_clauses = []
            for pat in patterns:
                lp = pat.pattern_str
                escaped_lp = escape_like_pattern(lp)
                # Add '%' at the start if not anchored at the beginning
                if not pat.anchored_start and not escaped_lp.startswith('%'):
                    escaped_lp = '%' + escaped_lp
                # Add '%' at the end if not anchored at the end
                if not pat.anchored_end and not escaped_lp.endswith('%'):
                    escaped_lp = escaped_lp + '%'
                positive_clause = f"COL LIKE '{escaped_lp}'"
                # Add ESCAPE clause if backslash is present
                if '\\' in lp:
                    positive_clause += " ESCAPE '\\'"
                positive_clauses.append(positive_clause)
                # If there are exclusions for this pattern
                exclusions = pat.exclusions
                if exclusions:
                    # Generate NOT LIKE patterns
                    positions = [e['position'] for e in exclusions]
                    chars_list = [e['chars'] for e in exclusions]
                    # Generate all combinations of excluded characters
                    for excluded_chars in product(*chars_list):
                        excluded_pattern = list(lp)
                        for pos, char in zip(positions, excluded_chars):
                            char = escape_literal_char(char)  # Escape literals here
                            excluded_pattern[pos] = char
                        negative_pattern = ''.join(excluded_pattern)
                        escaped_negative_pattern = escape_like_pattern(negative_pattern)
                        # Add '%' at the start if not anchored at the beginning
                        if not pat.anchored_start and not escaped_negative_pattern.startswith('%'):
                            escaped_negative_pattern = '%' + escaped_negative_pattern
                        # Add '%' at the end if not anchored at the end
                        if not pat.anchored_end and not escaped_negative_pattern.endswith('%'):
                            escaped_negative_pattern = escaped_negative_pattern + '%'
                        negative_clause = f"COL NOT LIKE '{escaped_negative_pattern}'"
                        # Add ESCAPE clause if backslash is present
                        if '\\' in negative_pattern:
                            negative_clause += " ESCAPE '\\'"
                        negative_clauses.append(negative_clause)
            # Remove duplicates
            negative_clauses = list(set(negative_clauses))

            # Combine positive clauses with OR, negative clauses with AND
            if positive_clauses and negative_clauses:
                like_clause = f"(({ ' OR '.join(positive_clauses) }) AND { ' AND '.join(negative_clauses) })"
            elif positive_clauses:
                like_clause = f"({ ' OR '.join(positive_clauses) })"
            elif negative_clauses:
                like_clause = f"({ ' AND '.join(negative_clauses) })"
            else:
                continue  # Should not happen

            like_clauses.append(like_clause)
        else:
            all_translated = False
            regexp_clauses.append(f"REGEXP_LIKE(COL, '{pi}')")

    if all_translated and like_clauses:
        final_query = ' OR '.join(like_clauses)
        translation_status = 'complete translation'
    elif any_translated:
        final_query = ' OR '.join(like_clauses + regexp_clauses)
        translation_status = 'partial translation'
    else:
        final_query = ' OR '.join(regexp_clauses)
        translation_status = 'not possible to translate to LIKE'

    return final_query, translation_status

# Example usage
if __name__ == "__main__":
    # Test pattern with end-of-string anchor
    pattern = 'ab[xy]21|r_e[^ab]7'
    final_query, translation_status = regex_to_like(pattern)
    print('Final Query:', final_query)
    print('Translation Status:', translation_status)
