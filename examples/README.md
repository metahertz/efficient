# examples

The Claude Code integration previously documented here (hook scripts,
settings.json, CLAUDE.md guidance, install script) now ships as a Claude Code
plugin — see the "Claude Code integration" section of the root README:

```
/plugin marketplace add metahertz/efficient
/plugin install efficient@efficient
```

The hook scripts themselves live in `plugin/scripts/` and can still be wired
manually into `.claude/settings.json` if you prefer not to use the plugin.
