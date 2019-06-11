"""Standard command line flag types."""

BoolProvider = provider(fields = ["value"])

IntProvider = provider(fields = ["value"])

StringProvider = provider(fields = ["value"])

def _bool_flag_impl(ctx):
  return BoolProvider(value = ctx.build_setting_value)

bool_flag = rule(
  implementation = _bool_flag_impl,
  build_setting = config.bool(flag = True),
)

def _int_flag_impl(ctx):
  return IntProvider(value = ctx.build_setting_value)

int_flag = rule(
  implementation = _int_flag_impl,
  build_setting = config.int(flag = True),
)

def _string_flag_impl(ctx):
  return StringProvider(value = ctx.build_setting_value)

string_flag = rule(
  implementation = _string_flag_impl,
  build_setting = config.string(flag = True),
)
