# Copyright 2015 The Bazel Authors. All rights reserved.
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

SASS_FILETYPES = FileType([".sass", ".scss"])

def collect_transitive_sources(ctx):
    source_files = set(order="compile")
    for dep in ctx.attr.deps:
        source_files += dep.transitive_sass_files
    return source_files

def _sass_library_impl(ctx):
    transitive_sources = collect_transitive_sources(ctx)
    transitive_sources += SASS_FILETYPES.filter(ctx.files.srcs)
    return struct(
        files = set(),
        transitive_sass_files = transitive_sources)

def _sass_binary_impl(ctx):
    # Reference the sass compiler and define the default options
    # that sass_binary uses.
    sassc = ctx.file._sassc
    options = [
        "--style={0}".format(ctx.attr.output_style),
        "--sourcemap",
    ]

    # Load up all the transitive sources as dependent includes.
    transitive_sources = collect_transitive_sources(ctx)
    for src in transitive_sources:
        options += ["-I={0}".format(src)]

    ctx.action(
        inputs = [sassc, ctx.file.src] + list(transitive_sources),
        executable = sassc,
        arguments = options + [ctx.file.src.path, ctx.outputs.css_file.path],
        mnemonic = "SassCompiler",
        outputs = [ctx.outputs.css_file, ctx.outputs.css_map_file],
    )

sass_deps_attr = attr.label_list(
    providers = ["transitive_sass_files"],
    allow_files = False,
)

sass_library = rule(
    implementation = _sass_library_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = SASS_FILETYPES,
            non_empty = True,
            mandatory = True,
        ),
        "deps": sass_deps_attr,
    },
)

sass_binary = rule(
    implementation = _sass_binary_impl,
    attrs = {
        "src": attr.label(
            allow_files = SASS_FILETYPES,
            mandatory = True,
            single_file = True,
        ),
        "output_style": attr.string(default = "compressed"),
        "deps": sass_deps_attr,
        "_sassc": attr.label(
            default = Label("//tools/build_defs/sass:sassc"),
            executable = True,
            single_file = True,
        ),
    },
    outputs = {
        "css_file": "%{name}.css",
        "css_map_file": "%{name}.css.map",
    },
)

LIBSASS_BUILD_FILE = """
package(default_visibility = ["@sassc//:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob([
         "src/**/*.h*",
         "src/**/*.c*",
    ]),
)

# Includes directive may seem unnecessary here, but its needed for the weird
# interplay between libsass/sassc projects. This is intentional.
cc_library(
    name = "headers",
    includes = ["include"],
    hdrs = glob(["include/**/*.h"]),
)
"""

SASSC_BUILD_FILE = """
package(default_visibility = ["//tools/build_defs/sass:__pkg__"])

cc_binary(
    name = "sassc",
    srcs = [
        "@libsass//:srcs",
        "sassc.c",
        "sassc_version.h",
],
    linkopts = ["-ldl", "-lm"],
    deps = ["@libsass//:headers"],
)
"""

def sass_repositories():
  native.new_http_archive(
      name = "libsass",
      url = "https://github.com/sass/libsass/archive/3.3.0-beta1.tar.gz",
      sha256 = "6a4da39cc0b585f7a6ee660dc522916f0f417c890c3c0ac7ebbf6a85a16d220f",
      build_file_content = LIBSASS_BUILD_FILE,
      strip_prefix = "libsass-3.3.0-beta1",
  )

  native.new_http_archive(
      name = "sassc",
      url = "https://github.com/sass/sassc/archive/3.3.0-beta1.tar.gz",
      sha256 = "87494218eea2441a7a24b40f227330877dbba75c5fa9014ac6188711baed53f6",
      build_file_content = SASSC_BUILD_FILE,
      strip_prefix = "sassc-3.3.0-beta1",
  )
