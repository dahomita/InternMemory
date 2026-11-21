# Agent Harness Reliability

An agent harness should enforce bounded tool calls and a maximum number of
execution steps. These limits prevent infinite tool loops.

Tool results require schema validation before they enter model context.
Malformed output should become a visible tool error instead of being silently
accepted as evidence.
