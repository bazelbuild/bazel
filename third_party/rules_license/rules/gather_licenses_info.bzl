# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rules and macros for collecting LicenseInfo providers."""

load(
    "@rules_license//rules:providers.bzl",
    "LicenseInfo",
    "LicensesInfo",
)

# Debugging verbosity
_VERBOSITY = 0

def _debug(loglevel, msg):
    if _VERBOSITY > loglevel:
        print(msg)  # buildifier: disable=print

def _get_transitive_licenses(deps, licenses, trans):
    for dep in deps:
        if LicenseInfo in dep:
            license = dep[LicenseInfo]
            _debug(1, "  depends on license: %s" % license.rule)
            licenses.append(license)
        if LicensesInfo in dep:
            license_list = dep[LicensesInfo].licenses
            if license_list:
                _debug(1, "  transitively depends on: %s" % licenses)
                trans.append(license_list)

def _gather_licenses_info_impl(target, ctx):
    licenses = []
    trans = []
    if hasattr(ctx.rule.attr, "applicable_licenses"):
        _get_transitive_licenses(ctx.rule.attr.applicable_licenses, licenses, trans)
    if hasattr(ctx.rule.attr, "deps"):
        _get_transitive_licenses(ctx.rule.attr.deps, licenses, trans)
    if hasattr(ctx.rule.attr, "srcs"):
        _get_transitive_licenses(ctx.rule.attr.srcs, licenses, trans)
    return [LicensesInfo(licenses = depset(tuple(licenses), transitive = trans))]

gather_licenses_info = aspect(
    doc = """Collects LicenseInfo providers into a single LicensesInfo provider.""",
    implementation = _gather_licenses_info_impl,
    attr_aspects = ["applicable_licenses", "deps", "srcs"],
    apply_to_generating_rules = True,
)

def write_licenses_info(ctx, deps, json_out):
    """Writes LicensesInfo providers for a set of targets as JSON.

    TODO(aiuto): Document JSON schema.

    Usage:
      write_licenses_info must be called from a rule implementation, where the
      rule has run the gather_licenses_info aspect on its deps to collect the
      transitive closure of LicenseInfo providers into a LicenseInfo provider.

      foo = rule(
        implementation = _foo_impl,
        attrs = {
           "deps": attr.label_list(aspects = [gather_licenses_info])
        }
      )

      def _foo_impl(ctx):
        ...
        out = ctx.actions.declare_file("%s_licenses.json" % ctx.label.name)
        write_licenses_info(ctx, ctx.attr.deps, licenses_file)

    Args:
      ctx: context of the caller
      deps: a list of deps which should have LicensesInfo providers.
            This requires that you have run the gather_licenses_info
            aspect over them
      json_out: output handle to write the JSON info
    """

    rule_template = """  {{
    "rule": "{rule}",
    "license_kinds": [{kinds}
    ],
    "copyright_notice": "{copyright_notice}",
    "package_name": "{package_name}",
    "license_text": "{license_text}"\n  }}"""

    kind_template = """
      {{
        "target": "{kind_path}",
        "name": "{kind_name}",
        "conditions": {kind_conditions}
      }}"""

    licenses = []
    for dep in deps:
        if LicensesInfo in dep:
            for license in dep[LicensesInfo].licenses.to_list():
                _debug(0, "  Requires license: %s" % license)
                kinds = []
                for kind in license.license_kinds:
                    kinds.append(kind_template.format(
                        kind_name = kind.name,
                        kind_path = kind.label,
                        kind_conditions = kind.conditions,
                    ))
                licenses.append(rule_template.format(
                    rule = license.rule,
                    copyright_notice = license.copyright_notice,
                    package_name = license.package_name,
                    license_text = license.license_text.path,
                    kinds = ",\n".join(kinds),
                ))
    ctx.actions.write(
        output = json_out,
        content = "[\n%s\n]\n" % ",\n".join(licenses),
    )
