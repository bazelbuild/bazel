"""Deprecation flag for packaging rules.

# TODO(aiuto): This should be a standard. 
"""

BoolProvider = provider(fields = ["value"])

def _bool_flag_impl(ctx):
  return BoolProvider(value = ctx.build_setting_value)

bool_flag = rule(
  implementation = _bool_flag_impl,
  build_setting = config.bool(flag = True),
)
