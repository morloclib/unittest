import sys

RED = '\033[0;31m'
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
RESET = '\033[0m'

def _good(msg):
    return f"{GREEN}{msg}{RESET}"

def _bad(msg):
    return f"{RED}{msg}{RESET}"

def _info(msg):
    return f"{BLUE}{msg}{RESET}"

# numpy and pyarrow are only consulted when a value already came from them, so
# the harness never forces either on a suite that does not use them.
def _module(name):
    return sys.modules.get(name)

def _is_ndarray(x):
    np = _module("numpy")
    return np is not None and isinstance(x, np.ndarray)

def _is_record_batch(x):
    pa = _module("pyarrow")
    return pa is not None and isinstance(x, pa.RecordBatch)

def _show(x):
    if _is_record_batch(x):
        return {name: x.column(i).to_pylist() for i, name in enumerate(x.column_names)}
    if _is_ndarray(x):
        return x.tolist()
    return repr(x)

# Structural equality. Record batches compare on ordered column names and
# per-column values. Arrays compare on shape, then elementwise: floating arrays
# within numpy's default tolerance, boxed arrays recursively, the rest exactly.
def _equal(x, y):
    if _is_record_batch(x) or _is_record_batch(y):
        if not (_is_record_batch(x) and _is_record_batch(y)):
            return False
        if list(x.column_names) != list(y.column_names) or x.num_rows != y.num_rows:
            return False
        return all(x.column(i).to_pylist() == y.column(i).to_pylist()
                   for i in range(x.num_columns))
    if _is_ndarray(x) or _is_ndarray(y):
        np = _module("numpy")
        xa = np.asarray(x)
        ya = np.asarray(y)
        if xa.shape != ya.shape:
            return False
        if xa.dtype == object or ya.dtype == object:
            return all(_equal(xa.flat[i], ya.flat[i]) for i in range(xa.size))
        if np.issubdtype(xa.dtype, np.floating) or np.issubdtype(ya.dtype, np.floating):
            return bool(np.allclose(xa, ya))
        return bool(np.array_equal(xa, ya))
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        return len(x) == len(y) and all(_equal(a, b) for a, b in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_equal(x[k], y[k]) for k in x)
    return x == y

def _record(msg, ok, details, results):
    (nfails, ntests) = results
    if ok:
        print(f"  {msg} ... {_good('PASS')}")
        return (nfails, ntests + 1)
    print(f"  {msg} ... {_bad('FAIL')}")
    for line in details():
        print(f"    {line}")
    return (nfails + 1, ntests + 1)

def testEqual(msg, x, y, results):
    return _record(msg, _equal(x, y),
        lambda: [f"expected: {_show(y)}", f"got:      {_show(x)}"], results)

def testTrue(msg, x, results):
    return _record(msg, x is True, lambda: [f"got: {x!r}"], results)

def testFalse(msg, x, results):
    return _record(msg, x is False, lambda: [f"got: {x!r}"], results)

def testNear(msg, tol, x, y, results):
    return _record(msg, abs(x - y) <= tol,
        lambda: [f"expected: {y!r} +/- {tol!r}", f"got:      {x!r}"], results)

def testGe(msg, x, y, results):
    return _record(msg, x >= y,
        lambda: [f"expected: >= {y!r}", f"got:         {x!r}"], results)

def testInRange(msg, x, lo, hi, results):
    return _record(msg, lo <= x <= hi,
        lambda: [f"expected: {lo!r} <= x <= {hi!r}", f"got:      {x!r}"], results)

def testLength(msg, xs, n, results):
    return _record(msg, len(xs) == n,
        lambda: [f"expected length: {n}", f"got length:      {len(xs)}"], results)

def testNonEmpty(msg, xs, results):
    return _record(msg, len(xs) > 0,
        lambda: [f"expected non-empty, got: {xs!r}"], results)

def testContains(msg, xs, x, results):
    return _record(msg, any(_equal(x, y) for y in xs),
        lambda: [f"expected {x!r} in {xs!r}"], results)

def testSubset(msg, xs, ys, results):
    missing = [x for x in xs if not any(_equal(x, y) for y in ys)]
    return _record(msg, not missing,
        lambda: [f"not in superset: {missing!r}"], results)

def testSameElements(msg, xs, ys, results):
    return _record(msg, sorted(xs) == sorted(ys),
        lambda: [f"expected sorted: {sorted(ys)!r}", f"got sorted:      {sorted(xs)!r}"],
        results)

def testNoDuplicates(msg, xs, results):
    dupes = sorted({x for x in xs if xs.count(x) > 1})
    return _record(msg, not dupes, lambda: [f"duplicates: {dupes!r}"], results)

def printGroup(msg, x):
    print(_info(msg))
    return x

def printSummary(results):
    (nfails, ntests) = results
    if nfails == 0:
        print(_good(f"All {ntests} tests pass"))
        return results
    raise RuntimeError(_bad(f"{nfails}/{ntests} tests failed"))
