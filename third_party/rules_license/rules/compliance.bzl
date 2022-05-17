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

"""Proof of concept. License compliance checking."""

load(
    "@rules_license//rules:gather_licenses_info.bzl",
    "gather_licenses_info",
    "write_licenses_info",
)
load(
    "@rules_license//rules:providers.bzl",
    "LicensesInfo",
)

# Debugging verbosity
_VERBOSITY = 0

def _debug(loglevel, msg):
    if _VERBOSITY > loglevel:
        print(msg)  # buildifier: disable=print

def _check_license_impl(ctx):
    # Gather all licenses and write information to one place

    _debug(0, "Check license: %s" % ctx.label)

    licenses_file = ctx.actions.declare_file("_%s_licenses_info.json" % ctx.label.name)
    write_licenses_info(ctx, ctx.attr.deps, licenses_file)

    license_files = []
    if ctx.outputs.license_texts:
        for dep in ctx.attr.deps:
            if LicensesInfo in dep:
                for license in dep[LicensesInfo].licenses.to_list():
                    license_files.append(license.license_text)

    # Now run the checker on it
    inputs = [licenses_file]
    outputs = [ctx.outputs.report]
    args = ctx.actions.args()
    args.add("--licenses_info", licenses_file.path)
    args.add("--report", ctx.outputs.report.path)
    if ctx.attr.check_conditions:
        args.add("--check_conditions")
    if ctx.outputs.copyright_notices:
        args.add("--copyright_notices", ctx.outputs.copyright_notices.path)
        outputs.append(ctx.outputs.copyright_notices)
    if ctx.outputs.license_texts:
        args.add("--license_texts", ctx.outputs.license_texts.path)
        outputs.append(ctx.outputs.license_texts)
        inputs.extend(license_files)
    ctx.actions.run(
        mnemonic = "CheckLicenses",
        progress_message = "Checking license compliance for %s" % ctx.label,
        inputs = inputs,
        outputs = outputs,
        executable = ctx.executable._checker,
        arguments = [args],
    )
    outputs.append(licenses_file)  # also make the json file available.
    return [DefaultInfo(files = depset(outputs))]

_check_license = rule(
    implementation = _check_license_impl,
    attrs = {
        "check_conditions": attr.bool(default = True, mandatory = False),
        "copyright_notices": attr.output(mandatory = False),
        "deps": attr.label_list(
            aspects = [gather_licenses_info],
        ),
        "license_texts": attr.output(mandatory = False),
        "report": attr.output(mandatory = True),
        "_checker": attr.label(
            default = Label("@rules_license//tools:checker_demo"),
            executable = True,
            allow_files = True,
            cfg = "host",
        ),
    },
)

def check_license(**kwargs):
    _check_license(**kwargs)

def _licenses_used_impl(ctx):
    """Gather all licenses and make it available as JSON."""
    write_licenses_info(ctx, ctx.attr.deps, ctx.outputs.out)
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

_licenses_used = rule(
    implementation = _licenses_used_impl,
    doc = """Internal tmplementation method for licenses_used().""",
    attrs = {
        "deps": attr.label_list(
            doc = """List of targets to collect LicenseInfo for.""",
            aspects = [gather_licenses_info],
        ),
        "out": attr.output(
            doc = """Output file.""",
            mandatory = True,
        ),
    },
)

def licenses_used(name, deps, out = None, **kwargs):
    """Collects LicensedInfo providers for a set of targets and writes as JSON.

    The output is a single JSON array, with an entry for each license used.
    See gather_licenses_info.bzl:write_licenses_info() for a description of the schema.

    Args:
      name: The target.
      deps: A list of targets to get LicenseInfo for. The output is the union of
            the result, not a list of information for each dependency.
      out: The output file name. Default: <name>.json.
      **kwargs: Other args

    Usage:

      licenses_used(
          name = "license_info",
          deps = [":my_app"],
          out = "license_info.json",
      )
    """
    if not out:
        out = name + ".json"
    _licenses_used(name = name, deps = deps, out = out, **kwargs)
