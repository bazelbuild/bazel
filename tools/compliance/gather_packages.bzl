# Copyright 2023 Google LLC
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

load("@rules_license//rules:filtered_rule_kinds.bzl", "aspect_filters")
load(
    "@rules_license//rules:providers.bzl",
    "LicenseInfo",
    "PackageInfo",
)
load(
    "@rules_license//rules/private:gathering_providers.bzl",
    "LicensedTargetInfo",
)
load("@rules_license//rules_gathering:trace.bzl", "TraceInfo")

load(":user_filtered_rule_kinds.bzl", "user_aspect_filters")

TransitivePackageInfo = provider(
    """Tostt""",
    fields = {
        "top_level_target": "Label: The top level target label we are examining.",
        "license_info": "depset(LicenseInfo)",
        "package_info": "depset(PackageInfo)",
        "packages": "depset(label)",

        "target_under_license": "Label: A target which will be associated with some licenses.",
        "traces": "list(string) - diagnostic for tracing a dependency relationship to a target.",
    },
)


def should_traverse(ctx, attr):
    """Checks if the dependent attribute should be traversed.

    Args:
      ctx: The aspect evaluation context.
      attr: The name of the attribute to be checked.

    Returns:
      True iff the attribute should be traversed.
    """
    k = ctx.rule.kind
    for filters in [aspect_filters, user_aspect_filters]:
        always_ignored = filters.get("*", [])
        if k in filters:
            attr_matches = filters[k]
            if (attr in attr_matches or
                "*" in attr_matches or
                ("_*" in attr_matches and attr.startswith("_")) or
                attr in always_ignored):
                return False

            for m in attr_matches:
                if attr == m:
                    return False
    return True


def _get_transitive_metadata(ctx, trans_license_info, trans_package_info, trans_packages, traces, provider, filter_func):
    """Pulls the transitive data up from attributes we care about.

    Collapses all the TransitivePackageInfo providers from my deps into lists
    that we can then turn into a single TPI.
    """
    attrs = [a for a in dir(ctx.rule.attr)]
    for name in attrs:
        if not filter_func(ctx, name):
            continue
        a = getattr(ctx.rule.attr, name)

        # Make anything singleton into a list for convenience.
        if type(a) != type([]):
            a = [a]
        for dep in a:
            # Ignore anything that isn't a target
            if type(dep) != "Target":
                continue

            # Targets can also include things like input files that won't have the
            # aspect, so we additionally check for the aspect rather than assume
            # it's on all targets.  Even some regular targets may be synthetic and
            # not have the aspect. This provides protection against those outlier
            # cases.
            if provider in dep:
                info = dep[provider]
                if info.license_info:
                    trans_license_info.append(info.license_info)
                if info.package_info:
                    trans_package_info.append(info.package_info)
                if info.packages:
                    trans_packages.append(info.packages)
                #if info.traces:
                #    for trace in info.traces:
                #        traces.append("(" + ", ".join([str(ctx.label), ctx.rule.kind, name]) + ") -> " + trace)


def gather_package_common(target, ctx, provider_factory, null_provider, metadata_providers, filter_func):
    """Collect license and other metadata info from myself and my deps.

    Any single target might directly depend on a license or package_info, or depend on
    something that transitively depends on a one of those, or neither.
    This aspect bundles all those into a single provider. At each level, we add
    in new direct license deps found and forward up the transitive information
    collected so far.

    This is a common abstraction for crawling the dependency graph. It is
    parameterized to allow specifying the provider that is populated with
    results. It is configurable to select only a subset of providers. It
    is also configurable to specify which dependency edges should not
    be traced for the purpose of tracing the graph.

    NOTE: This is highly based on @rules_license//rules/licenses_common.bzl
    So it may look more complex than it needs to be. That is because I am
    trying to work out some more genearl patterns here and then backport
    to rules_license.

    Args:
      target: The target of the aspect.
      ctx: The aspect evaluation context.
      provider_factory: abstracts the provider returned by this aspect
      metadata_providers: a list of other providers of interest
      filter_func: a function that returns true iff the dep edge should be ignored

    Returns:
      provider of parameterized type
    """

    # A hack until https://github.com/bazelbuild/rules_license/issues/89 is
    # fully resolved. If exec is in the bin_dir path, then the current
    # configuration is probably cfg = exec.
    if "-exec-" in ctx.bin_dir.path:
        return [null_provider]

    # First we gather my direct license attachments
    licenses = []
    package_info = []
    packages = None
    if ctx.rule.kind == "_license":
        # Don't try to gather licenses from the license rule itself. We'll just
        # blunder into the text file of the license and pick up the default
        # attribute of the package, which we don't want.
        pass
    else:
        if hasattr(ctx.rule.attr, "applicable_licenses"):
            for dep in ctx.rule.attr.applicable_licenses:
                if LicenseInfo in dep:
                    licenses.append(dep[LicenseInfo])
                if PackageInfo in dep:
                    package_info.depend(dep[LicenseInfo])

    # Record all the external repos anyway.
    target_name = str(target.label)
    if target_name.startswith('@') and target_name[1] != '/':
        packages = [target.label]
        # print(str(target.label))

    # Now gather transitive collection of providers from the targets
    # this target depends upon.
    trans_license_info = []
    trans_package_info = []
    trans_packages = []
    traces = []
    _get_transitive_metadata(ctx, trans_license_info, trans_package_info, trans_packages, traces, provider_factory, filter_func)

    if (not licenses
        and not package_info
        and not trans_license_info
        and not trans_package_info
        and not trans_packages):
        return [null_provider]

    # If this is the target, start the sequence of traces.
    if ctx.attr._trace[TraceInfo].trace and ctx.attr._trace[TraceInfo].trace in str(ctx.label):
        traces = [ctx.attr._trace[TraceInfo].trace]

    # Trim the number of traces accumulated since the output can be quite large.
    # A few representative traces are generally sufficient to identify why a dependency
    # is incorrectly incorporated.
    if len(traces) > 10:
        traces = traces[0:10]

    """
    if licenses:
        # At this point we have a target and a list of directly used licenses.
        # Bundle those together so we can report the exact targets that cause the
        # dependency on each license. Since a list cannot be stored in a
        # depset, even inside a provider, the list is concatenated into a
        # string and will be unconcatenated in the output phase.
        direct_license_uses = [LicensedTargetInfo(
            target_under_license = target.label,
            licenses = ",".join([str(x.label) for x in licenses]),
        )]
    else:
        direct_license_uses = None
    """

    return [provider_factory(
        top_level_target = target.label,
        # target_under_license = target.label,
        license_info = depset(direct = licenses, transitive = trans_license_info),
        package_info = depset(direct = package_info, transitive = trans_package_info),
        packages = depset(direct = packages, transitive = trans_packages),
        traces = traces,
    )]

# Singleton instance of nothing present. Each level of the aspect must return something,
# so we use one to save space instead of making a lot of empty ones.
_NULL_INFO = TransitivePackageInfo(license_info = depset(), package_info = depset(), packages=depset())

def _gather_package_impl(target, ctx):
    ret = gather_package_common(
        target = target,
        ctx = ctx,
        provider_factory = TransitivePackageInfo,
        null_provider = _NULL_INFO,
        metadata_providers = [PackageInfo],
        filter_func = should_traverse)
    return ret

gather_package_info = aspect(
    doc = """Collects License and Package providers into a single TransitivePackageInfo provider.""",
    implementation = _gather_package_impl,
    attr_aspects = ["*"],
    attrs = {
        "_trace": attr.label(default = "@rules_license//rules:trace_target"),
    },
    provides = [TransitivePackageInfo],
    apply_to_generating_rules = True,
)

def _write_package_info_impl(target, ctx):
    """Write transitive package info into a JSON file

    Args:
      target: The target of the aspect.
      ctx: The aspect evaluation context.

    Returns:
      OutputGroupInfo
    """
    if not TransitivePackageInfo in target:
        # buildifier: disable=print
        print("Missing package_info for", target)
        return [OutputGroupInfo(package_info = depset())]
    info = target[TransitivePackageInfo]
    outs = []

    #XXX # If the result doesn't contain licenses, we simply return the provider
    #if not hasattr(info, "target_under_license"):
    #    return [OutputGroupInfo(package_info = depset())]

    # Write the output file for the target
    name = "%s_package_info.json" % ctx.label.name
    content = "[\n%s\n]\n" % ",\n".join(package_info_to_json(info))
    out = ctx.actions.declare_file(name)
    ctx.actions.write(
        output = out,
        content = content,
    )
    print("============== wrote", name)
    outs.append(out)
    if ctx.attr._trace[TraceInfo].trace:
        trace = ctx.actions.declare_file("%s_trace_info.json" % ctx.label.name)
        ctx.actions.write(output = trace, content = "\n".join(info.traces))
        outs.append(trace)

    return [OutputGroupInfo(package_info = depset(outs))]

gather_package_info_and_write = aspect(
    doc = """Collects TransitiveMetadataInfo providers and writes JSON representation to a file.

    Usage:
      bazel build //some:target \
          --aspects=...:gather_package_info.bzl%gather_package_info_and_write
          --output_groups=package_info
    """,
    implementation = _write_package_info_impl,
    attrs = {
        "_trace": attr.label(default = "@rules_license//rules:trace_target"),
    },
    provides = [OutputGroupInfo],
    requires = [gather_package_info],
    apply_to_generating_rules = True,
)

###############################################################################
#
# When we finish figuring this out, the code below should move to a new source
# file.

def _strip_null_repo(label):
    """Removes the null repo name (e.g. @//) from a string.

    The is to make str(label) compatible between bazel 5.x and 6.x
    """
    s = str(label)
    if s.startswith('@//'):
        return s[1:]
    elif s.startswith('@@//'):
        return s[2:]
    return s

def _bazel_package(label):
    clean_label = _strip_null_repo(label)
    return clean_label[0:-(len(label.name) + 1)]

def package_info_to_json(package_info):
    """Render a TransitivePackageInfo to JSON

    schema:
      top_level_target: name of the target
      licenses: list of license target data
      package_info: list of package_info
      packages: list of remote packages we depend on

    Args:
      package_info: A TransitivePackageInfo.

    Returns:
      [(str)] list of TransitivePackageInfo values rendered as JSON.
    """

    # licenses: List of license() targets
    # package_info: List of labels of remote packages we depend on
    # packages: List of labels of remote packages we depend on
    main_template = """  {{
    "top_level_target": "{top_level_target}",
    "licenses": [{licenses}
    ],
    "package_info": [{package_info}
    ],
    "packages": [{packages}
    ]\n  }}"""

    dep_template = """
      {{
        "target_under_license": "{target_under_license}",
        "licenses": [
          {licenses}
        ]
      }}"""

    license_template = """
      {{
        "label": "{label}",
        "bazel_package": "{bazel_package}",
        "license_kinds": [{kinds}
        ],
        "copyright_notice": "{copyright_notice}",
        "package_name": "{package_name}",
        "package_url": "{package_url}",
        "package_version": "{package_version}",
        "license_text": "{license_text}",
      }}"""

    kind_template = """
          {{
            "target": "{kind_path}",
            "name": "{kind_name}",
            "conditions": {kind_conditions}
          }}"""

    package_info_template = """
          {{
            "target": "{label}",
            "bazel_package": "{bazel_package}",
            "package_name": "{package_name}",
            "package_url": "{package_url}",
            "package_version": "{package_version}"
          }}"""

    if not hasattr(package_info, "top_level_target"):
        return ""

    all_licenses = []
    for license in sorted(package_info.license_info.to_list(), key = lambda x: x.label):
        kinds = []
        for kind in sorted(license.license_kinds, key = lambda x: x.name):
            kinds.append(kind_template.format(
                kind_name = kind.name,
                kind_path = kind.label,
                kind_conditions = kind.conditions,
            ))

        if license.license_text:
            # Special handling for synthetic LicenseInfo
            text_path = (license.license_text.package + "/" + license.license_text.name if type(license.license_text) == "Label" else license.license_text.path)
            all_licenses.append(license_template.format(
                copyright_notice = license.copyright_notice,
                kinds = ",".join(kinds),
                license_text = text_path,
                package_name = license.package_name,
                package_url = license.package_url,
                package_version = license.package_version,
                label = _strip_null_repo(license.label),
                bazel_package =  _bazel_package(license.label),
            ))

    all_package_info = []
    for package in sorted(package_info.package_info.to_list(), key = lambda x: x.label):
        all_package_info.append(package_info_template.format(
            label = _strip_null_repo(package.label),
            package_name = package.package_name,
            package_url = package.package_url,
            package_version = package.package_version,
        ))

    all_packages = []
    for mi in sorted(package_info.packages.to_list(), key = lambda x: str(x)):
        all_packages.append('"%s"' % str(mi))

    return [main_template.format(
        top_level_target = _strip_null_repo(package_info.top_level_target),
        licenses = ",".join(all_licenses),
        package_info = ",".join(all_package_info),
        packages = ",".join(all_packages),
    )]
