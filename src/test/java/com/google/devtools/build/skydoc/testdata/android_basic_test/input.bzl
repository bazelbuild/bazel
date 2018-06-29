def exercise_the_api():
    var1 = android_common.create_device_broker_info("")
    var2 = ApkInfo
    var3 = AndroidInstrumentationInfo
    var4 = AndroidDeviceBrokerInfo
    var5 = AndroidResourcesInfo
    var6 = AndroidNativeLibsInfo

exercise_the_api()

def my_rule_impl(ctx):
    return struct()


android_related_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule does android-related things.",
    attrs = {
        "first": attr.label(mandatory = True, allow_files = True, single_file = True),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, mandatory = False),
    },
)
