"""Rules for building Bazel's native-image server."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "CPP_LINK_EXECUTABLE_ACTION_NAME", "C_COMPILE_ACTION_NAME")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

_MAIN_CLASS = "com.google.devtools.build.lib.bazel.Bazel"

# Bazel's server image needs generated native-image configs and the
# libmanagement_ext.so side output, which the public rules_graalvm native_image
# rule cannot currently declare.
def _resolve_cc_toolchain(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    c_compiler_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
    )
    compile_variables = cc_common.create_compile_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
    )
    compile_env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
        variables = compile_variables,
    )
    compile_requirements = cc_common.get_execution_requirements(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
    )
    link_variables = cc_common.create_link_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
    )
    link_env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
        variables = link_variables,
    )
    link_requirements = cc_common.get_execution_requirements(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
    )

    env = {}
    env.update(compile_env)
    env.update(link_env)

    path_set = {}
    for tool_path in [c_compiler_path]:
        tool_dir, _, _ = tool_path.rpartition("/")
        if tool_dir:
            path_set[tool_dir] = None
    if "/bin" not in path_set:
        path_set["/bin"] = None
    if "/usr/bin" not in path_set:
        path_set["/usr/bin"] = None
    env["PATH"] = ctx.configuration.host_path_separator.join(sorted(path_set.keys()))

    return struct(
        c_compiler_path = c_compiler_path,
        env = env,
        execution_requirements = compile_requirements + link_requirements,
        files = cc_toolchain.all_files,
    )

def _bazel_server_native_image_impl(ctx):
    classpath = depset([ctx.file.deploy_jar])

    generated_reflection_configs = [
        ctx.actions.declare_file(ctx.attr.name + "/" + basename)
        for basename in [
            "caffeine_reflect_config.json",
            "proto_reflect_config.json",
            "options_reflect_config.json",
            "converter_reflect_config.json",
            "rule_reflect_config.json",
            "starlark_reflect_config.json",
            "bazel_methods_reflect_config.json",
            "bazel_fields_reflect_config.json",
            "gson_reflect_config.json",
            "bzlmod_reflect_config.json",
            "netty_jctools_reflect_config.json",
            "netty_buffer_reflect_config.json",
        ]
    ]
    dynamic_proxy_config = ctx.actions.declare_file(ctx.attr.name + "/dynamic_proxy_config.json")

    config_args = ctx.actions.args()
    config_args.add(ctx.file.deploy_jar)
    config_args.add(dynamic_proxy_config)
    config_args.add_all(generated_reflection_configs)
    ctx.actions.run(
        executable = ctx.executable._config_generator,
        arguments = [config_args],
        inputs = [ctx.file.deploy_jar],
        tools = [ctx.executable._config_generator],
        outputs = [dynamic_proxy_config] + generated_reflection_configs,
        mnemonic = "BazelNativeImageConfig",
        progress_message = "Generating native-image configs %{label}",
    )

    binary = ctx.actions.declare_file(ctx.attr.executable_name)
    jdk_library = ctx.actions.declare_file("libmanagement_ext.so")
    build_output_json = ctx.actions.declare_file(ctx.attr.name + "/build-output.json")
    bundle = ctx.actions.declare_file(ctx.attr.name + "/native-image.nib")

    graalvm_files = ctx.attr.graalvm_files[DefaultInfo].files
    cc_toolchain = _resolve_cc_toolchain(ctx)

    direct_inputs = [
        ctx.file.jni_configuration,
        ctx.file.reflection_configuration,
        dynamic_proxy_config,
    ] + generated_reflection_configs

    args = ctx.actions.args().use_param_file("@%s", use_always = False)
    args.add("-Dfile.encoding=ISO-8859-1")
    args.add("-Dnative.encoding=UTF-8")
    args.add("--add-opens=java.base/java.lang=ALL-UNNAMED")
    args.add("--add-exports=java.base/jdk.internal.misc=ALL-UNNAMED")
    args.add("--enable-native-access=ALL-UNNAMED")
    args.add("--enable-monitoring=threaddump")
    args.add("-H:+UnlockExperimentalVMOptions")
    args.add("-H:+AddAllCharsets")
    args.add("-H:DefaultCharset=ISO-8859-1")
    if ctx.attr.optimization:
        args.add("-O" + ctx.attr.optimization)
    if ctx.attr.gc:
        args.add("--gc=" + ctx.attr.gc)
    args.add(bundle, format = "--bundle-create=%s,dry-run")
    args.add(build_output_json, format = "-H:BuildOutputJSONFile=%s")
    args.add_joined(
        [ctx.file.reflection_configuration] + generated_reflection_configs,
        join_with = ",",
        format_joined = "-H:ReflectionConfigurationFiles=%s",
    )
    args.add(ctx.file.jni_configuration, format = "-H:JNIConfigurationFiles=%s")
    args.add(dynamic_proxy_config, format = "-H:DynamicProxyConfigurationFiles=%s")
    args.add(ctx.attr.include_resources, format = "-H:IncludeResources=%s")
    args.add("-march=x86-64-v2")
    args.add(cc_toolchain.c_compiler_path, format = "--native-compiler-path=%s")
    if ctx.attr.parallelism > 0:
        args.add(ctx.attr.parallelism, format = "--parallelism=%s")
    args.add("-o")
    args.add(binary)
    args.add_joined("-cp", classpath, join_with = ctx.configuration.host_path_separator)
    args.add(
        "--initialize-at-run-time=com.google.devtools.build.lib.profiler.SystemNetworkStatsServiceImpl,com.google.devtools.build.lib.unix.ProcessUtilsServiceImpl,com.google.devtools.build.lib.util.StringEncoding,io.grpc.netty,io.netty.bootstrap,io.netty.buffer,io.netty.channel,io.netty.handler,io.netty.internal.tcnative,io.netty.util.NetUtil,io.netty.util.ResourceLeakDetector,io.netty.util.concurrent.AbstractScheduledEventExecutor,io.netty.util.internal.MacAddressUtil,sun.nio.fs.Util"
    )
    args.add(
        "--initialize-at-build-time=" + ",".join([
            "com.google.devtools.build.lib.bazel.repository.decompressor.CompressedTarFunction$MarkedIso88591Charset",
            "io.netty.util.internal.shaded.org.jctools",
        ])
    )
    args.add(ctx.attr.main_class)

    env = dict(cc_toolchain.env)
    env.update({
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
    })

    native_image_inputs = depset(
        direct = direct_inputs,
        transitive = [
            classpath,
            graalvm_files,
            cc_toolchain.files,
        ],
    )
    native_image_apply_inputs = depset(
        direct = [bundle],
        transitive = [
            graalvm_files,
            cc_toolchain.files,
        ],
    )

    ctx.actions.run(
        executable = ctx.executable.native_image_tool,
        arguments = [args],
        inputs = native_image_inputs,
        outputs = [bundle],
        env = env,
        execution_requirements = {requirement: "" for requirement in cc_toolchain.execution_requirements},
        mnemonic = "BazelNativeImageBundle",
        progress_message = "Creating Bazel native-image bundle %{label}",
        toolchain = "@bazel_tools//tools/cpp:toolchain_type",
    )

    apply_args = ctx.actions.args()
    apply_args.add(ctx.executable.native_image_tool)
    apply_args.add(bundle)
    apply_args.add(ctx.attr.executable_name)
    apply_args.add(binary)
    apply_args.add(jdk_library)
    apply_args.add(build_output_json)

    ctx.actions.run_shell(
        command = """
set -eu

execroot="$PWD"
native_image_tool="$execroot/$1"
bundle="$execroot/$2"
executable_name="$3"
binary="$4"
jdk_library="$5"
build_output_json="$6"
image_output="$execroot/$(dirname "$build_output_json")/native-image.output"

chmod -R u+w "$image_output" 2>/dev/null || true
rm -rf "$image_output"
trap 'rm -rf "$image_output"' EXIT

"$native_image_tool" "--bundle-apply=$bundle" -o "$executable_name"

image_source="$image_output/default/$executable_name"
if [ ! -f "$image_source" ]; then
  echo "native-image bundle did not produce $image_source" >&2
  find "$image_output" -maxdepth 3 -type f >&2 || true
  exit 1
fi
mv "$image_source" "$binary"
chmod a+x "$binary"

management_source=""
for candidate in "$image_output/default/libmanagement_ext.so" "$image_output/other/libmanagement_ext.so"; do
  if [ -f "$candidate" ]; then
    management_source="$candidate"
    break
  fi
done
if [ -z "$management_source" ]; then
  echo "native-image bundle did not produce libmanagement_ext.so" >&2
  find "$image_output" -maxdepth 3 -type f >&2 || true
  exit 1
fi
mv "$management_source" "$jdk_library"

build_output_source=""
for candidate in "$image_output/other/build-output.json" "$image_output/default/build-output.json"; do
  if [ -f "$candidate" ]; then
    build_output_source="$candidate"
    break
  fi
done
if [ -z "$build_output_source" ]; then
  echo "native-image bundle did not produce build-output.json" >&2
  find "$image_output" -maxdepth 3 -type f >&2 || true
  exit 1
fi
mv "$build_output_source" "$build_output_json"
""",
        arguments = [apply_args],
        inputs = native_image_apply_inputs,
        tools = [ctx.executable.native_image_tool],
        outputs = [binary, jdk_library, build_output_json],
        env = env,
        execution_requirements = {requirement: "" for requirement in cc_toolchain.execution_requirements},
        mnemonic = "BazelNativeImage",
        progress_message = "Building Bazel native-image server %{label}",
        toolchain = "@bazel_tools//tools/cpp:toolchain_type",
    )

    return [
        DefaultInfo(
            executable = binary,
            files = depset([binary, jdk_library]),
        ),
        OutputGroupInfo(
            native_image_diagnostics = depset([build_output_json]),
            native_image_bundle = depset([bundle]),
            native_image_output = depset([binary, jdk_library, build_output_json]),
        ),
    ]

bazel_server_native_image = rule(
    implementation = _bazel_server_native_image_impl,
    attrs = {
        "deploy_jar": attr.label(
            allow_single_file = [".jar"],
            mandatory = True,
        ),
        "executable_name": attr.string(
            default = "A-server-native",
        ),
        "graalvm_files": attr.label(
            mandatory = True,
        ),
        "include_resources": attr.string(
            default = ".*",
        ),
        "jni_configuration": attr.label(
            allow_single_file = [".json"],
            mandatory = True,
        ),
        "main_class": attr.string(
            default = _MAIN_CLASS,
        ),
        "native_image_tool": attr.label(
            allow_files = True,
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "optimization": attr.string(
            default = "2",
            values = [
                "b",
                "s",
                "0",
                "1",
                "2",
                "3",
            ],
        ),
        "gc": attr.string(
            values = [
                "",
                "G1",
                "epsilon",
                "serial",
            ],
        ),
        "parallelism": attr.int(),
        "reflection_configuration": attr.label(
            allow_single_file = [".json"],
            mandatory = True,
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_config_generator": attr.label(
            default = Label("//src:bazel_server_native_image_configs"),
            cfg = "exec",
            executable = True,
        ),
    },
    executable = True,
    fragments = [
        "cpp",
        "platform",
    ],
    toolchains = [
        "@bazel_tools//tools/cpp:toolchain_type",
    ],
)
