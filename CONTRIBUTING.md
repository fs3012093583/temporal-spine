# Contributing

Contributions are welcome through focused pull requests.

Before submitting a change:

1. preserve the query-independent User-Spine;
2. preserve monotonic, non-repeated component loading;
3. keep the request envelope stable across phases;
4. do not add benchmark data, private histories, credentials, or generated model answers;
5. run `pytest -q`.

Please separate protocol changes from benchmark integrations so that invariants remain reviewable.
