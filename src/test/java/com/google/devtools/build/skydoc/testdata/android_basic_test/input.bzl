def exercise_the_api():
    _var1 = android_common.create_device_broker_info("")
    _var2 = ApkInfo
    _var3 = AndroidInstrumentationInfo
    _var4 = AndroidDeviceBrokerInfo
    _var5 = AndroidResourcesInfo
    _var6 = AndroidNativeLibsInfo
    _var7 = AndroidSdkInfo
    _var8 = android_data

exercise_the_api()

def my_rule_impl(ctx):
    return struct()

android_related_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule does android-related things.",
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, mandatory = False),
    },
)
