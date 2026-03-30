load("@rules_java//java:defs.bzl", "java_library")

########################################################################
# Description:
#   Main Java code for Bazel
#
# IMPORTANT:  NOT A PUBLIC INTERFACE.
#
# DO NOT ADD A DEPENDENCY on any targets or sub-packages outside of
# __subpackages__ without first talking to Bazel maintainers.
#
# Here are two good reasons:
#
# 1.  FRAGILITY. This is not a stable API, it's an internal interface.
#     It changes from time to time, and your client *will* be broken.
#
# 2.  CORRECTNESS.  The BUILD file parser changes rapidly.  If
#     your application depends upon it, your application will start to
#     bit-rot almost immediately.
#
# There are better alternatives.  The 'bazel query' tool has
# stable command-line and programmatic interfaces that will work with any
# version of Bazel.  They are also debugged and documented, so you'll
# avoid many common traps and pitfalls when (e.g.)  computing
# dependencies.
package(
    default_visibility = ["//src:__subpackages__"],
)

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/main/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis:srcs",
        "//src/main/java/com/google/devtools/build/lib/authandtls:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel:srcs",
        "//src/main/java/com/google/devtools/build/lib/bugreport:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildtool:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildtool/buildevent:srcs",
        "//src/main/java/com/google/devtools/build/lib/causes:srcs",
        "//src/main/java/com/google/devtools/build/lib/clock:srcs",
        "//src/main/java/com/google/devtools/build/lib/cmdline:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:srcs",
        "//src/main/java/com/google/devtools/build/lib/concurrent:srcs",
        "//src/main/java/com/google/devtools/build/lib/dynamic:srcs",
        "//src/main/java/com/google/devtools/build/lib/events:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec:srcs",
        "//src/main/java/com/google/devtools/build/lib/graph:srcs",
        "//src/main/java/com/google/devtools/build/lib/io:srcs",
        "//src/main/java/com/google/devtools/build/lib/jni:srcs",
        "//src/main/java/com/google/devtools/build/lib/metrics:srcs",
        "//src/main/java/com/google/devtools/build/lib/network:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages/metrics:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages/producers:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics:srcs",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:srcs",
        "//src/main/java/com/google/devtools/build/lib/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/common:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/compat:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/engine:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/query/aspectresolvers:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote:srcs",
        "//src/main/java/com/google/devtools/build/lib/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime/events:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime/mobileinstall:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:srcs",
        "//src/main/java/com/google/devtools/build/lib/server:srcs",
        "//src/main/java/com/google/devtools/build/lib/shell:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/module:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/server:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkprofiler:srcs",
        "//src/main/java/com/google/devtools/build/lib/supplier:srcs",
        "//src/main/java/com/google/devtools/build/lib/testing/common:srcs",
        "//src/main/java/com/google/devtools/build/lib/testing/vfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/unix:srcs",
        "//src/main/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/main/java/com/google/devtools/build/lib/util:srcs",
        "//src/main/java/com/google/devtools/build/lib/versioning:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/common/options:srcs",
    ],
    visibility = [
        "//src:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

java_library(
    name = "all-build-rules",
    exports = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/rules:legacy_exports",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

########################################################################
#
# Documentation generation
#

filegroup(
    name = "generated_docs",
    srcs = [
        "//src/main/java/com/google/devtools/build/lib:gen_buildencyclopedia",
        "//src/main/java/com/google/devtools/build/lib:gen_command-line-reference",
        "//src/main/java/com/google/devtools/build/lib:gen_skylarklibrary",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = [
        "//src/main/java/com/google/devtools/build/lib/analysis:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildtool:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec:srcs",
        "//src/main/java/com/google/devtools/build/lib/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime:srcs",
    ],
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [
        ":docs_embedded_in_sources",
        "//src/main/java/com/google/devtools/build/docgen:bazel_link_map",
        "//src/main/starlark/docgen:gen_be_proto_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_java_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_cpp_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_objc_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_python_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_shell_stardoc_proto",
    ],
    outs = ["build-encyclopedia.zip"],
    cmd = (
        "mkdir -p $(@D)/be && " +
        "$(location //src/main/java/com/google/devtools/build/docgen:docgen_bin)" +
        " --link_map_path=$(location //src/main/java/com/google/devtools/build/docgen:bazel_link_map)" +
        " --provider=com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider" +
        " --input_root=$$PWD" +
        " --input_dir=$$PWD/src/main/java/com/google/devtools/build/lib" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_proto_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_java_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_cpp_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_objc_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_python_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_shell_stardoc_proto)" +
        " --output_dir=$(@D)/be" +
        " --create_toc=true &&" +
        "zip -qj $@ $(@D)/be/*"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_bin",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
    ],
    visibility = ["//site:__pkg__"],
)

genrule(
    name = "gen_command-line-reference",
    srcs = [
        "//site:command-line-reference-prefix.html",
        "//site:command-line-reference-suffix.html",
    ],
    outs = ["command-line-reference.html"],
    cmd = (
        "cat $(location //site:command-line-reference-prefix.html) > $@ && " +
        "TMP=$$(mktemp -d $${TMPDIR:-/tmp}/tmp.XXXXXXXXXX) && " +
        # TODO(#11179): Remove when the stub template fix is released.
        "export JARBIN=$(JAVABASE)/bin/jar && " +
        "$(location //src/main/java/com/google/devtools/build/lib/bazel:BazelServer) " +
        "--batch " +
        "--install_base=$${TMP} --output_base=$${TMP}/output/ --output_user_root=$${TMP} " +
        "--failure_detail_out=$${TMP}/output/failure_detail.rawproto " +
        "help everything-as-html >> $@ 2>$${TMP}/stderr && " +
        "cat $(location //site:command-line-reference-suffix.html) >> $@"
    ),
    toolchains = ["@rules_java//toolchains:current_host_java_runtime"],
    tools = [
        "//src/main/java/com/google/devtools/build/lib/bazel:BazelServer",
    ],
    visibility = ["//site:__pkg__"],
)

genrule(
    name = "gen_skylarklibrary",
    srcs = ["//src/main/java/com/google/devtools/build/docgen:bazel_link_map"],
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location //src/main/java/com/google/devtools/build/docgen:skydoc_bin)" +
          " $(@D)/skylark-lib" +
          " --link_map_path=$(location //src/main/java/com/google/devtools/build/docgen:bazel_link_map)" +
          " --create_toc=true" +
          " && OUTPUT=$$PWD/$@" +
          " && pushd $(@D)/skylark-lib" +
          " && zip -qr $$OUTPUT ." +
          " && popd",
    tools = ["//src/main/java/com/google/devtools/build/docgen:skydoc_bin"],
    visibility = ["//site:__pkg__"],
)

genrule(
    name = "gen_reference_docs",
    srcs = [
        ":build-encyclopedia.zip",
        ":command-line-reference.html",
        ":skylark-library.zip",
        "//tools/build_defs/repo:doc",
    ],
    outs = ["reference-docs.zip"],
    cmd = "mkdir -p $(@D)/gen_docs/reference/be && " +
          "cp $(location :command-line-reference.html) $(@D)/gen_docs/reference/ && " +
          "unzip -q -d $(@D)/gen_docs/reference/be/ $(location :build-encyclopedia.zip) && " +
          "mkdir -p $(@D)/gen_docs/rules/lib/repo && " +
          "unzip -q -d $(@D)/gen_docs/rules/lib/ $(location :skylark-library.zip) && " +
          "tar -xf $(location //tools/build_defs/repo:doc) -C $(@D)/gen_docs/rules/lib/repo/ && " +
          # Ugly hack: we want to preserve paths relative to gen_docs/,
          # which means we have to temporarily cd into that directory and zip with -r.
          "pushd $(@D)/gen_docs && " +
          "zip -qr tmp.zip . && " +
          "popd && mv $(@D)/gen_docs/tmp.zip $@",
    visibility = ["//scripts/docs:__pkg__"],
)

# TODO: replace gen_reference_docs with this target once the docs migration is done.
genrule(
    name = "gen_mdx_reference_docs",
    srcs = [
        ":build-encyclopedia.zip",
        ":command-line-reference.html",
        ":skylark-library.zip",
        "//tools/build_defs/repo:doc",
    ],
    outs = ["mdx-reference-docs.zip"],
    cmd = "mkdir -p $(@D)/html_for_mdx/reference/be && " +
          "cp $(location :command-line-reference.html) $(@D)/html_for_mdx/reference/ && " +
          "unzip -q -d $(@D)/html_for_mdx/reference/be/ $(location :build-encyclopedia.zip) && " +
          "mkdir -p $(@D)/html_for_mdx/rules/lib/repo && " +
          "unzip -q -d $(@D)/html_for_mdx/rules/lib/ $(location :skylark-library.zip) && " +
          "tar -xf $(location //tools/build_defs/repo:doc) -C $(@D)/html_for_mdx/rules/lib/repo/ && " +
          "mkdir -p $(@D)/mdx_docs && " +
          "$(location //scripts/docs:docs2mdx) --in_dir $(@D)/html_for_mdx --out_dir $(@D)/mdx_docs && " +
          # Ugly hack: we want to preserve paths relative to mdx_docs/,
          # which means we have to temporarily cd into that directory and zip with -r.
          "pushd $(@D)/mdx_docs && " +
          "zip -qr tmp.zip . && " +
          "popd && mv $(@D)/mdx_docs/tmp.zip $@",
    tools = [
        "//scripts/docs:docs2mdx",
    ],
    visibility = ["//scripts/docs:__pkg__"],
)

genrule(
    name = "gen_api_proto",
    srcs = [
        "//src/main/java/com/google/devtools/build/docgen:bazel_link_map",
        "//src/main/starlark/docgen:gen_be_proto_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_java_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_cpp_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_objc_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_python_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_shell_stardoc_proto",
        ":docs_embedded_in_sources",
    ],
    outs = ["builtin.pb"],
    cmd = (
        "$(location //src/main/java/com/google/devtools/build/docgen:api_exporter)" +
        " --output_file=$@" +
        " --link_map_path=$(location //src/main/java/com/google/devtools/build/docgen:bazel_link_map) " +
        " --provider=com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider" +
        " --input_root=$$PWD" +
        " --input_dir=$$PWD/src/main/java/com/google/devtools/build/lib" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_proto_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_java_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_cpp_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_objc_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_python_stardoc_proto)" +
        " --be_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_shell_stardoc_proto)"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:api_exporter",
    ],
)
