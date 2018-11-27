def exercise_the_api():
    _var6 = configuration_field("foo", "bar")

exercise_the_api()

def _build_setting_impl(ctx):
    return []

string_flag = rule(
    implementation = _build_setting_impl,
    build_setting = config.string(flag = True),
)

int_setting = rule(
    implementation = _build_setting_impl,
    build_setting = config.int(flag = False),
)
