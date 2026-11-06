"""Final-answer extraction; never match arbitrary intermediate numbers.

No unrestricted symbolic evaluation. Numeric expressions use a small, bounded
arithmetic AST and exact rational arithmetic; symbolic answers use normalization.
"""
import ast
import re
from fractions import Fraction

ANSWER_CHECKER_VERSION = "structured-commas-exact-integer-decimal-atol1e-6-v2"
DECIMAL_ABS_TOLERANCE = Fraction(1, 10**6)
_DECIMAL_LITERAL = re.compile(r"(?:\d\.\d*|\.\d+|\d[eE][+-]?\d)")
_GROUPED_SCALAR = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?%?")


def _boxed(text):
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                values.append((match.start(), text[start:end]))
                break
    return values


def extract_final(text):
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = _boxed(text)
    pattern = r"(?:the\s+(?:final\s+)?answer\s+is|final\s+answer|answer|\u7b54\u6848)\s*[:：]?\s*(.+)"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        candidates.append((match.start(), match.group(1)))
    for match in re.finditer(r"####\s*(.+)", text):
        candidates.append((match.start(), match.group(1)))
    if candidates:
        answer = max(candidates, key=lambda item: item[0])[1].strip()
        if answer.startswith(r"\boxed"):
            boxes = _boxed(answer)
            if boxes:
                answer = boxes[-1][1]
        return answer.strip().strip("$ ").rstrip(".。") or None
    last = text.strip().splitlines()[-1].strip().strip("$ ").rstrip(".。")
    if "=" in last:
        last = last.rsplit("=", 1)[1].strip().rstrip(".。")
    if re.fullmatch(r"[\d\s.,+*/()^%{}\\\-]+", last) and re.search(r"\d", last):
        return last
    return None


def normalize(value):
    value = value.strip().strip("$ ").rstrip(".。")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\pi", "pi").replace(r"\cdot", "*").replace(r"\times", "*")
    value = value.replace(r"\,", "").replace(r"\!", "")
    value = value.replace("−", "-").replace(r"\%", "%")
    for _ in range(12):
        changed = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", value)
        if changed == value:
            break
        value = changed
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value.replace(",", "") if _GROUPED_SCALAR.fullmatch(value) else value


def _number(value):
    if len(value) > 128:
        return None
    if value.endswith("%"):
        inner = _number(value[:-1])
        return inner / 100 if inner is not None else None
    try:
        expression = value.replace("^", "**")
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 40:
            return None

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return Fraction(node.value)
            if isinstance(node, ast.Constant) and type(node.value) is float:
                literal = ast.get_source_segment(expression, node)
                exponent = re.search(r"[eE]([+-]?[\d_]+)$", literal)
                if exponent and abs(int(exponent.group(1).replace("_", ""))) > 128:
                    raise ValueError("numeric exponent exceeds parser bounds")
                return Fraction(literal)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 12:
                    return left ** int(right)
            raise ValueError("not bounded numeric arithmetic")

        return evaluate(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return None


def equivalent(first, second):
    if first is None or second is None:
        return False
    first, second = normalize(first), normalize(second)
    if first == second:
        return bool(first)
    left, right = _number(first), _number(second)
    if left is None or right is None:
        return False
    if left == right:
        return True
    if left.denominator == right.denominator == 1:
        return False
    if not (_DECIMAL_LITERAL.search(first) or _DECIMAL_LITERAL.search(second)):
        return False
    return abs(left - right) <= DECIMAL_ABS_TOLERANCE
