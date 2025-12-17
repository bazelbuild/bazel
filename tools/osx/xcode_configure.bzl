# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Repository rule to generate host xcode_config and xcode_version targets.

   The xcode_config and xcode_version targets are configured for xcodes/SDKs
   installed on the local host.
"""

OSX_EXECUTE_TIMEOUT = 600

# SDK platform definitions for generating xcode_sdk_variant targets.
# Each tuple: (sdk_key, config_setting_name, platform_dir, supported_targets_key)
# - sdk_key: Used for target naming and identifying the SDK
# - config_setting_name: The config_setting in @apple_support//xcode/private
# - platform_dir: Directory name under Platforms/ (e.g., "iPhoneOS")
# - supported_targets_key: Key in SDKSettings.json's SupportedTargets dict
_SDK_PLATFORMS = [
    ("iphoneos", "is_iphoneos", "iPhoneOS", "iphoneos"),
    ("iphonesimulator", "is_iphonesimulator", "iPhoneSimulator", "iphonesimulator"),
    ("macosx", "is_macos", "MacOSX", "macosx"),
    ("appletvos", "is_appletvos", "AppleTVOS", "appletvos"),
    ("appletvsimulator", "is_appletvsimulator", "AppleTVSimulator", "appletvsimulator"),
    ("watchos", "is_watchos", "WatchOS", "watchos"),
    ("watchsimulator", "is_watchsimulator", "WatchSimulator", "watchsimulator"),
    ("xros", "is_xros", "XROS", "xros"),
    ("xrsimulator", "is_xrsimulator", "XRSimulator", "xrsimulator"),
]

def _search_string(fullstring, prefix, suffix):
    """Returns the substring between two given substrings of a larger string.

    Args:
      fullstring: The larger string to search.
      prefix: The substring that should occur directly before the returned string.
      suffix: The substring that should occur directly after the returned string.
    Returns:
      A string occurring in fullstring exactly prefixed by prefix, and exactly
      terminated by suffix. For example, ("hello goodbye", "lo ", " bye") will
      return "good". If there is no such string, returns the empty string.
    """

    prefix_index = fullstring.find(prefix)
    if (prefix_index < 0):
        return ""
    result_start_index = prefix_index + len(prefix)
    suffix_index = fullstring.find(suffix, result_start_index)
    if (suffix_index < 0):
        return ""
    return fullstring[result_start_index:suffix_index]

def _parse_sdk_settings(repository_ctx, developer_dir, platform_dir, supported_targets_key):
    """Parses SDKSettings.json for a specific SDK platform.

    Args:
        repository_ctx: Repository context for executing commands.
        developer_dir: Path to Xcode's Developer directory.
        platform_dir: Platform directory name (e.g., "iPhoneOS").
        supported_targets_key: Key in SupportedTargets (e.g., "iphoneos").

    Returns:
        A struct with SDK settings, or None if the SDK doesn't exist or can't be parsed.
    """
    sdk_settings_path = "{}/Platforms/{}.platform/Developer/SDKs/{}.sdk/SDKSettings.json".format(
        developer_dir,
        platform_dir,
        platform_dir,
    )

    settings = json.decode(repository_ctx.read(sdk_settings_path))
    if not settings:
        fail("Unable to read SDK information from: {}".format(sdk_settings_path))

    supported_targets = settings.get("SupportedTargets", {})
    sdk_info = supported_targets.get(supported_targets_key, {})

    if not sdk_info:
        fail("Supported targets not found for SDK: {} with target key: {}".format(sdk_settings_path, supported_targets_key))

    # Transform DeviceFamilies from list of dicts to dict of name -> identifier
    device_families = {}
    for family in sdk_info.get("DeviceFamilies", []):
        name = family.get("Name", "")
        identifier = family.get("Identifier", "")
        if name and identifier:
            device_families[name] = identifier

    return struct(
        archs = sdk_info.get("Archs", []),
        clangrt_name = sdk_info.get("ClangRuntimeLibraryPlatformName", ""),
        device_families = device_families,
        llvm_triple_environment = sdk_info.get("LLVMTargetTripleEnvironment", ""),
        llvm_triple_os = sdk_info.get("LLVMTargetTripleSys", ""),
        llvm_triple_vendor = sdk_info.get("LLVMTargetTripleVendor", ""),
        maximum_supported_os_version = sdk_info.get("MaximumDeploymentTarget", ""),
        minimum_supported_os_version = sdk_info.get("MinimumDeploymentTarget", ""),
        minimum_swift_concurrency_in_os_version = sdk_info.get("SwiftConcurrencyMinimumDeploymentTarget", ""),
        minimum_swift_in_os_version = sdk_info.get("SwiftOSRuntimeMinimumDeploymentTarget", ""),
        platform_directory_name = platform_dir,
        version = settings.get("Version", ""),
    )

def _generate_sdk_variant_target(name, sdk_settings, build_version):
    """Generates an xcode_sdk_variant target definition string.

    Args:
        name: Target name for the xcode_sdk_variant.
        sdk_settings: Struct returned from _parse_sdk_settings.
        build_version: Xcode build version string.

    Returns:
        String containing the xcode_sdk_variant target definition.
    """
    archs_str = "[%s]" % ", ".join(["'%s'" % a for a in sdk_settings.archs])

    families_entries = ["'%s': '%s'" % (k, v) for k, v in sdk_settings.device_families.items()]
    families_str = "{%s}" % ", ".join(families_entries)

    return """xcode_sdk_variant(
  name = '{name}',
  archs = {archs},
  build_version = '{build_version}',
  clangrt_name = '{clangrt_name}',
  device_families = {device_families},
  llvm_triple_environment = '{llvm_triple_environment}',
  llvm_triple_os = '{llvm_triple_os}',
  llvm_triple_vendor = '{llvm_triple_vendor}',
  maximum_supported_os_version = '{maximum_supported_os_version}',
  minimum_supported_os_version = '{minimum_supported_os_version}',
  minimum_swift_concurrency_in_os_version = '{minimum_swift_concurrency_in_os_version}',
  minimum_swift_in_os_version = '{minimum_swift_in_os_version}',
  platform_directory_name = '{platform_directory_name}',
  platform_name = '{platform_name}',
  resources_platform_name = '{resources_platform_name}',
  version = '{version}',
)
""".format(
        name = name,
        archs = archs_str,
        build_version = build_version,
        clangrt_name = sdk_settings.clangrt_name,
        device_families = families_str,
        llvm_triple_environment = sdk_settings.llvm_triple_environment,
        llvm_triple_os = sdk_settings.llvm_triple_os,
        llvm_triple_vendor = sdk_settings.llvm_triple_vendor,
        maximum_supported_os_version = sdk_settings.maximum_supported_os_version,
        minimum_supported_os_version = sdk_settings.minimum_supported_os_version,
        minimum_swift_concurrency_in_os_version = sdk_settings.minimum_swift_concurrency_in_os_version,
        minimum_swift_in_os_version = sdk_settings.minimum_swift_in_os_version,
        platform_directory_name = sdk_settings.platform_directory_name,
        platform_name = sdk_settings.platform_directory_name,
        resources_platform_name = sdk_settings.platform_directory_name,
        version = sdk_settings.version,
    )

def _search_sdk_output(output, sdkname):
    """Returns the SDK version given xcodebuild stdout and an sdkname."""
    return _search_string(output, "(%s" % sdkname, ")")

def _xcode_version_output(repository_ctx, name, version, aliases, developer_dir, timeout):
    """Returns a string containing xcode_version and xcode_sdk_variant build targets."""
    build_contents = ""
    decorated_aliases = []
    error_msg = ""
    for alias in aliases:
        decorated_aliases.append("'%s'" % alias)
    repository_ctx.report_progress("Fetching SDK information for Xcode %s" % version)

    # Get SDK versions via xcodebuild -version -sdk
    xcodebuild_result = repository_ctx.execute(
        ["xcrun", "xcodebuild", "-version", "-sdk"],
        timeout,
        {"DEVELOPER_DIR": developer_dir},
    )
    if (xcodebuild_result.return_code != 0):
        error_msg = (
            "Invoking xcodebuild failed, developer dir: {devdir} ," +
            "return code {code}, stderr: {err}, stdout: {out}"
        ).format(
            devdir = developer_dir,
            code = xcodebuild_result.return_code,
            err = xcodebuild_result.stderr,
            out = xcodebuild_result.stdout,
        )

    # Get build version via xcodebuild -version
    xcode_build_version = ""
    xcodebuild_version_result = repository_ctx.execute(
        ["xcrun", "xcodebuild", "-version"],
        timeout,
        {"DEVELOPER_DIR": developer_dir},
    )
    if xcodebuild_version_result.return_code == 0:
        xcode_build_version = _search_string(
            xcodebuild_version_result.stdout,
            "Build version ",
            "\n",
        )

    ios_sdk_version = _search_sdk_output(xcodebuild_result.stdout, "iphoneos")
    tvos_sdk_version = _search_sdk_output(xcodebuild_result.stdout, "appletvos")
    macos_sdk_version = _search_sdk_output(xcodebuild_result.stdout, "macosx")
    visionos_sdk_version = _search_sdk_output(xcodebuild_result.stdout, "xros")
    watchos_sdk_version = _search_sdk_output(xcodebuild_result.stdout, "watchos")

    # Generate xcode_sdk_variant targets for each platform
    sdk_variant_targets = {}
    version_suffix = version.replace(".", "_")
    for sdk_key, config_setting, platform_dir, supported_targets_key in _SDK_PLATFORMS:
        sdk_settings = _parse_sdk_settings(
            repository_ctx,
            developer_dir,
            platform_dir,
            supported_targets_key,
        )
        if sdk_settings:
            sdk_target_name = "{}_{}".format(sdk_key, version_suffix)
            build_contents += _generate_sdk_variant_target(
                sdk_target_name,
                sdk_settings,
                xcode_build_version,
            )
            sdk_variant_targets[config_setting] = sdk_target_name

    # Generate xcode_version target
    build_contents += "xcode_version(\n  name = '%s'," % name
    build_contents += "\n  version = '%s'," % version
    if aliases:
        build_contents += "\n  aliases = [%s]," % ", ".join(decorated_aliases)
    if ios_sdk_version:
        build_contents += "\n  default_ios_sdk_version = '%s'," % ios_sdk_version
    if tvos_sdk_version:
        build_contents += "\n  default_tvos_sdk_version = '%s'," % tvos_sdk_version
    if macos_sdk_version:
        build_contents += "\n  default_macos_sdk_version = '%s'," % macos_sdk_version
    if visionos_sdk_version:
        build_contents += "\n  default_visionos_sdk_version = '%s'," % visionos_sdk_version
    if watchos_sdk_version:
        build_contents += "\n  default_watchos_sdk_version = '%s'," % watchos_sdk_version

    # Add sdk attribute with select() if we have any SDK variants
    if sdk_variant_targets:
        build_contents += "\n  sdk = select({"
        for config_setting, target_name in sorted(sdk_variant_targets.items()):
            build_contents += "\n    '@apple_support//xcode/private:%s': ':%s'," % (
                config_setting,
                target_name,
            )
        build_contents += "\n    '//conditions:default': '@apple_support//xcode/private:empty_sdk',"
        build_contents += "\n  }),"

    build_contents += "\n)\n"
    if error_msg:
        build_contents += "\n# Error: " + error_msg.replace("\n", " ") + "\n"
        print(error_msg)
    return build_contents

VERSION_CONFIG_STUB = """
load("@apple_support//xcode:xcode_config.bzl", "xcode_config")
xcode_config(name = 'host_xcodes')
"""

def run_xcode_locator(repository_ctx, xcode_locator_src_label):
    """Generates xcode-locator from source and runs it.

    Builds xcode-locator in the current repository directory.
    Returns the standard output of running xcode-locator with -v, which will
    return information about locally installed Xcode toolchains and the versions
    they are associated with.

    This should only be invoked on a darwin OS, as xcode-locator cannot be built
    otherwise.

    Args:
      repository_ctx: The repository context.
      xcode_locator_src_label: The label of the source file for xcode-locator.
    Returns:
      A 2-tuple containing:
      output: A list representing installed xcode toolchain information. Each
          element of the list is a struct containing information for one installed
          toolchain. This is an empty list if there was an error building or
          running xcode-locator.
      err: An error string describing the error that occurred when attempting
          to build and run xcode-locator, or None if the run was successful.
    """
    repository_ctx.report_progress("Building xcode-locator")
    xcodeloc_src_path = str(repository_ctx.path(xcode_locator_src_label))
    env = repository_ctx.os.environ
    if "BAZEL_OSX_EXECUTE_TIMEOUT" in env:
        timeout = int(env["BAZEL_OSX_EXECUTE_TIMEOUT"])
    else:
        timeout = OSX_EXECUTE_TIMEOUT

    xcrun_result = repository_ctx.execute([
        "env",
        "-i",
        "DEVELOPER_DIR={}".format(env.get("DEVELOPER_DIR", default = "")),
        "xcrun",
        "--sdk",
        "macosx",
        "clang",
        "-mmacosx-version-min=10.13",
        "-fobjc-arc",
        "-framework",
        "CoreServices",
        "-framework",
        "Foundation",
        "-o",
        "xcode-locator-bin",
        xcodeloc_src_path,
    ], timeout)

    if (xcrun_result.return_code != 0):
        suggestion = ""
        if "Agreeing to the Xcode/iOS license" in xcrun_result.stderr:
            suggestion = ("(You may need to sign the Xcode license." +
                          " Try running 'sudo xcodebuild -license')")
        error_msg = (
            "Generating xcode-locator-bin failed. {suggestion} " +
            "return code {code}, stderr: {err}, stdout: {out}"
        ).format(
            suggestion = suggestion,
            code = xcrun_result.return_code,
            err = xcrun_result.stderr,
            out = xcrun_result.stdout,
        )
        return ([], error_msg.replace("\n", " "))

    repository_ctx.report_progress("Running xcode-locator")
    xcode_locator_result = repository_ctx.execute(
        ["./xcode-locator-bin", "-v"],
        timeout,
    )
    if (xcode_locator_result.return_code != 0):
        error_msg = (
            "Invoking xcode-locator failed, " +
            "return code {code}, stderr: {err}, stdout: {out}"
        ).format(
            code = xcode_locator_result.return_code,
            err = xcode_locator_result.stderr,
            out = xcode_locator_result.stdout,
        )
        return ([], error_msg.replace("\n", " "))
    xcode_toolchains = []

    # xcode_dump is comprised of newlines with different installed Xcode versions,
    # each line of the form <version>:<comma_separated_aliases>:<developer_dir>.
    xcode_dump = xcode_locator_result.stdout
    for xcodeversion in xcode_dump.split("\n"):
        if ":" in xcodeversion:
            infosplit = xcodeversion.split(":")
            toolchain = struct(
                version = infosplit[0],
                aliases = infosplit[1].split(","),
                developer_dir = infosplit[2],
            )
            xcode_toolchains.append(toolchain)
    return (xcode_toolchains, None)

def _darwin_build_file(repository_ctx):
    """Evaluates local system state to create xcode_config and xcode_version targets."""
    repository_ctx.report_progress("Fetching the default Xcode version")
    env = repository_ctx.os.environ

    if "BAZEL_OSX_EXECUTE_TIMEOUT" in env:
        timeout = int(env["BAZEL_OSX_EXECUTE_TIMEOUT"])
    else:
        timeout = OSX_EXECUTE_TIMEOUT

    xcodebuild_result = repository_ctx.execute([
        "env",
        "-i",
        "DEVELOPER_DIR={}".format(env.get("DEVELOPER_DIR", default = "")),
        "xcrun",
        "xcodebuild",
        "-version",
    ], timeout)

    (toolchains, xcodeloc_err) = run_xcode_locator(
        repository_ctx,
        Label(repository_ctx.attr.xcode_locator),
    )

    if xcodeloc_err:
        return VERSION_CONFIG_STUB + "\n# Error: " + xcodeloc_err + "\n"

    default_xcode_version = ""
    default_xcode_build_version = ""
    if xcodebuild_result.return_code == 0:
        default_xcode_version = _search_string(xcodebuild_result.stdout, "Xcode ", "\n")
        default_xcode_build_version = _search_string(
            xcodebuild_result.stdout,
            "Build version ",
            "\n",
        )
    default_xcode_target = ""
    target_names = []
    buildcontents = """
load("@apple_support//xcode:available_xcodes.bzl", "available_xcodes")
load("@apple_support//xcode:xcode_config.bzl", "xcode_config")
load("@apple_support//xcode:xcode_sdk_variant.bzl", "xcode_sdk_variant")
load("@apple_support//xcode:xcode_version.bzl", "xcode_version")
"""

    for toolchain in toolchains:
        version = toolchain.version
        aliases = toolchain.aliases
        developer_dir = toolchain.developer_dir
        target_name = "version%s" % version.replace(".", "_")
        buildcontents += _xcode_version_output(
            repository_ctx,
            target_name,
            version,
            aliases,
            developer_dir,
            timeout,
        )
        target_label = "':%s'" % target_name
        target_names.append(target_label)
        if (version.startswith(default_xcode_version) and
            version.endswith(default_xcode_build_version)):
            default_xcode_target = target_label
    buildcontents += "xcode_config(\n  name = 'host_xcodes',"
    if target_names:
        buildcontents += "\n  versions = [%s]," % ", ".join(target_names)
    if not default_xcode_target and target_names:
        default_xcode_target = sorted(target_names, reverse = True)[0]
        print("No default Xcode version is set with 'xcode-select'; picking %s" %
              default_xcode_target)
    if default_xcode_target:
        buildcontents += "\n  default = %s," % default_xcode_target

    buildcontents += "\n)\n"
    buildcontents += "available_xcodes(\n  name = 'host_available_xcodes',"
    if target_names:
        buildcontents += "\n  versions = [%s]," % ", ".join(target_names)
    if default_xcode_target:
        buildcontents += "\n  default = %s," % default_xcode_target
    buildcontents += "\n)\n"
    if repository_ctx.attr.remote_xcode:
        buildcontents += "xcode_config(name = 'all_xcodes',"
        buildcontents += "\n  remote_versions = '%s', " % repository_ctx.attr.remote_xcode
        buildcontents += "\n  local_versions = ':host_available_xcodes', "
        buildcontents += "\n)\n"
    return buildcontents

def _impl(repository_ctx):
    """Implementation for the local_config_xcode repository rule.

    Generates a BUILD file containing a root xcode_config target named 'host_xcodes',
    which points to an xcode_version target for each version of Xcode installed on
    the local host machine. If no versions of Xcode are present on the machine
    (for instance, if this is a non-darwin OS), creates a stub target.

    Args:
      repository_ctx: The repository context.
    """

    os_name = repository_ctx.os.name
    build_contents = "package(default_visibility = ['//visibility:public'])\n\n"
    if (os_name.startswith("mac os")):
        build_contents += _darwin_build_file(repository_ctx)
    else:
        build_contents += VERSION_CONFIG_STUB
    repository_ctx.file("BUILD", build_contents)

xcode_autoconf = repository_rule(
    environ = [
        "DEVELOPER_DIR",
        "XCODE_VERSION",
    ],
    implementation = _impl,
    configure = True,
    attrs = {
        "xcode_locator": attr.string(),
        "remote_xcode": attr.string(),
    },
)

def xcode_configure(xcode_locator_label, remote_xcode_label = None):
    """Generates a repository containing host Xcode version information."""
    xcode_autoconf(
        name = "local_config_xcode",
        xcode_locator = xcode_locator_label,
        remote_xcode = remote_xcode_label,
    )

def _xcode_configure_extension_impl(module_ctx):
    xcode_configure("@bazel_tools//tools/osx:xcode_locator.m")
    return module_ctx.extension_metadata(reproducible = True)

xcode_configure_extension = module_extension(implementation = _xcode_configure_extension_impl)
