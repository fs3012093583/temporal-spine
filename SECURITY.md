# Security

Temporal-Spine restores historical user, assistant, and tool content into the model context. That
content may be stale, incorrect, or adversarial. Source-faithful restoration does **not** make the
content factually authoritative.

Production deployments should isolate tenants, encrypt stored history, enforce component-level
access control, and treat restored content as untrusted data rather than system instructions.

Please report vulnerabilities privately to `fanshun@stu.scu.edu.cn`. Do not include API keys,
private conversation logs, or personal data in a public issue.
