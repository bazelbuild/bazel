"""The input file for the providers for attributes test"""

load(":testdata/providers_for_attributes_test/dep.bzl", "DepProviderInfo")

def my_rule_impl(ctx):
    return []

MyProviderInfo = provider(
    fields = {
        "foo": "Something foo-related.",
        "bar": "Something bar-related.",
    },
)

OtherProviderInfo = provider()
other_provider_info = OtherProviderInfo(fields = ["foo"])

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule does things.",
    attrs = {
        "first": attr.label_keyed_string_dict(
            providers = [MyProviderInfo, PyInfo, cc_common.CcToolchainInfo, my_undefined_module.MyInfo],
            doc = "this is the first attribute.",
        ),
        "second": attr.label_list(
            providers = [[CcInfo], [OtherProviderInfo, DepProviderInfo]],
        ),
        "third": attr.label(
            providers = [OtherProviderInfo],
        ),
        "fourth": attr.label(
            providers = [ProtoInfo, DefaultInfo, JavaInfo, MyUndefinedInfo],
        ),
        "fifth": attr.label(
            providers = [["LegacyProvider", "ObjectProvider"], [DefaultInfo, JavaInfo]],
        ),
        "sixth": attr.label(
            providers = ["LegacyProvider"],
        ),
    },
)
