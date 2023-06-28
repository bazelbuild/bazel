# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""
Definition of JavaInfo provider.
"""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/java/java_plugin_info.bzl", "merge_without_outputs")
load(":common/cc/cc_info.bzl", "CcInfo")

# TODO(hvd): remove this when:
# - we have a general provider-type checking API
# - no longer need to check for --experimental_google_legacy_api
_java_common_internal = _builtins.internal.java_common_internal_do_not_use

_JavaOutputInfo = provider(
    doc = "The outputs of Java compilation.",
    fields = {
        "class_jar": "(File) A classes jar file.",
        "compile_jar": "(File) An interface jar file.",
        "ijar": "Deprecated: Please use compile_jar.",
        "compile_jdeps": "(File) Compile time dependencies information (deps.proto file).",
        "generated_class_jar": "(File) A jar containing classes generated via annotation processing.",
        "generated_source_jar": "(File) The source jar created as a result of annotation processing.",
        "native_headers_jar": "(File) A jar of CC header files supporting native method implementation.",
        "manifest_proto": "(File) The manifest protobuf file of the manifest generated from JavaBuilder.",
        "jdeps": "(File) The jdeps protobuf file of the manifest generated from JavaBuilder.",
        "source_jars": "([File]) A list of sources archive files.",
        "source_jar": "Deprecated: Please use source_jars instead.",
    },
)
_ModuleFlagsInfo = provider(
    doc = "Provider for the runtime classpath contributions of a Java binary.",
    fields = {
        "add_exports": "(depset[str]) Add-Exports configuration.",
        "add_opens": "(depset[str]) Add-Opens configuration.",
    },
)
_JavaRuleOutputJarsInfo = provider(
    doc = "Deprecated: use java_info.java_outputs. Information about outputs of a Java rule.",
    fields = {
        "jdeps": "Deprecated: Use java_info.java_outputs.",
        "native_headers": "Deprecated: Use java_info.java_outputs[i].jdeps.",
        "jars": "Deprecated: Use java_info.java_outputs[i].native_headers_jar.",
    },
)
_JavaGenJarsInfo = provider(
    doc = "Deprecated: Information about jars that are a result of annotation processing for a Java rule.",
    fields = {
        "enabled": "Deprecated. Returns true if annotation processing was applied on this target.",
        "class_jar": "Deprecated: Please use JavaInfo.java_outputs.generated_class_jar instead.",
        "source_jar": "Deprecated: Please use JavaInfo.java_outputs.generated_source_jar instead.",
        "transitive_class_jars": "Deprecated. A transitive set of class file jars from annotation " +
                                 "processing of this rule and its dependencies.",
        "transitive_source_jars": "Deprecated. A transitive set of source archives from annotation " +
                                  "processing of this rule and its dependencies.",
        "processor_classpath": "Deprecated: Please use JavaInfo.plugins instead.",
        "processor_classnames": "Deprecated: Please use JavaInfo.plugins instead.",
    },
)

def _validate_provider_list(provider_list, what, expected_provider_type):
    _java_common_internal.check_provider_instances(provider_list, what, expected_provider_type)

def _javainfo_init(
        output_jar,
        compile_jar,
        source_jar = None,
        compile_jdeps = None,
        generated_class_jar = None,
        generated_source_jar = None,
        native_headers_jar = None,
        manifest_proto = None,
        neverlink = False,
        deps = [],
        runtime_deps = [],
        exports = [],
        exported_plugins = [],
        jdeps = None,
        native_libraries = []):
    """The JavaInfo constructor

    Args:
        output_jar: (File) The jar that was created as a result of a compilation.
        compile_jar: (File) A jar that is the compile-time dependency in lieu of `output_jar`.
        source_jar: (File) The source jar that was used to create the output jar. Optional.
        compile_jdeps: (File) jdeps information about compile time dependencies to be consumed by
            JavaCompileAction. This should be a binary proto encoded using the deps.proto protobuf
            included with Bazel. If available this file is typically produced by a header compiler.
            Optional.
        generated_class_jar: (File) A jar file containing class files compiled from sources
            generated during annotation processing. Optional.
        generated_source_jar: (File) The source jar that was created as a result of annotation
            processing. Optional.
        native_headers_jar: (File) A jar containing CC header files supporting native method
            implementation (typically output of javac -h). Optional.
        manifest_proto: (File) Manifest information for the rule output (if available). This should
            be a binary proto encoded using the manifest.proto protobuf included with Bazel. IDEs
            and other tools can use this information for more efficient processing. Optional.
        neverlink: (bool) If true, only use this library for compilation and not at runtime.
        deps: ([JavaInfo]) Compile time dependencies that were used to create the output jar.
        runtime_deps: ([JavaInfo]) Runtime dependencies that are needed for this library.
        exports: ([JavaInfo]) Libraries to make available for users of this library.
        exported_plugins: ([JavaPluginInfo]) Optional. A list of exported plugins.
        jdeps: (File) jdeps information for the rule output (if available). This should be a binary
            proto encoded using the deps.proto protobuf included with Bazel. If available this file
            is typically produced by a compiler. IDEs and other tools can use this information for
            more efficient processing. Optional.
        native_libraries: ([CcInfo]) Native library dependencies that are needed for this library.

    Returns:
        (dict) arguments to the JavaInfo provider constructor
    """
    _validate_provider_list(deps, "deps", JavaInfo)
    _validate_provider_list(runtime_deps, "runtime_deps", JavaInfo)
    _validate_provider_list(exports, "exports", JavaInfo)
    _validate_provider_list(native_libraries, "native_libraries", CcInfo)

    source_jars = [source_jar] if source_jar else []
    plugin_info = merge_without_outputs(exported_plugins + exports)
    if neverlink:
        transitive_runtime_jars = depset()
    else:
        transitive_runtime_jars = depset(
            order = "preorder",
            direct = [output_jar],
            transitive = [dep.transitive_runtime_jars for dep in exports + deps + runtime_deps],
        )
    transitive_compile_time_jars = depset(
        order = "preorder",
        direct = [compile_jar] if compile_jar else [],
        transitive = [dep.transitive_compile_time_jars for dep in exports + deps],
    )
    java_outputs = [_JavaOutputInfo(
        class_jar = output_jar,
        compile_jar = compile_jar,
        ijar = compile_jar,  # deprecated
        compile_jdeps = compile_jdeps,
        generated_class_jar = generated_class_jar,
        generated_source_jar = generated_source_jar,
        native_headers_jar = native_headers_jar,
        manifest_proto = manifest_proto,
        jdeps = jdeps,
        source_jars = source_jars,
        source_jar = source_jar,  # deprecated
    )]

    result = {
        "transitive_runtime_jars": transitive_runtime_jars,
        "transitive_runtime_deps": transitive_runtime_jars,  # deprecated
        "transitive_compile_time_jars": transitive_compile_time_jars,
        "transitive_deps": transitive_compile_time_jars,  # deprecated
        "compile_jars": depset(
            order = "preorder",
            direct = [compile_jar] if compile_jar else [],
            transitive = [dep.compile_jars for dep in exports],
        ),
        "full_compile_jars": depset(
            order = "preorder",
            direct = [output_jar],
            transitive = [
                dep.full_compile_jars
                for dep in exports
            ],
        ),
        "source_jars": source_jars,
        "runtime_output_jars": [output_jar],
        "transitive_source_jars": depset(
            direct = source_jars,
            # TODO(hvd): native also adds source jars from deps, but this should be unnecessary
            transitive = [
                dep.transitive_source_jars
                for dep in deps + runtime_deps + exports
            ],
        ),
        "module_flags_info": _ModuleFlagsInfo(
            add_exports = depset(transitive = [
                dep.module_flags_info.add_exports
                for dep in deps + exports
            ]),
            add_opens = depset(transitive = [
                dep.module_flags_info.add_opens
                for dep in deps + exports
            ]),
        ),
        "plugins": plugin_info.plugins,
        "api_generating_plugins": plugin_info.api_generating_plugins,
        "java_outputs": java_outputs,
        # deprecated
        "outputs": _JavaRuleOutputJarsInfo(
            jars = java_outputs,
            jdeps = jdeps,
            native_headers = native_headers_jar,
        ),
        "annotation_processing": _JavaGenJarsInfo(
            enabled = False,
            class_jar = generated_class_jar,
            source_jar = generated_source_jar,
            transitive_class_jars = depset(
                direct = [generated_class_jar] if generated_class_jar else [],
                transitive = [
                    dep.annotation_processing.transitive_class_jars
                    for dep in deps + exports
                    if dep.annotation_processing
                ],
            ),
            transitive_source_jars = depset(
                direct = [generated_source_jar] if generated_source_jar else [],
                transitive = [
                    dep.annotation_processing.transitive_source_jars
                    for dep in deps + exports
                    if dep.annotation_processing
                ],
            ),
            processor_classnames = depset(),
            processor_classpath = depset(),
        ),
        "compilation_info": None,
        "_neverlink": neverlink,
        "_transitive_full_compile_time_jars": depset(
            order = "preorder",
            direct = [output_jar],
            transitive = [dep._transitive_full_compile_time_jars for dep in exports + deps],
        ),
        "_plugin_info": plugin_info,
        "_compile_time_java_dependencies": depset(
            order = "preorder",
            transitive = [dep._compile_time_java_dependencies for dep in exports] +
                         ([depset([compile_jdeps])] if compile_jdeps else []),
        ),
    }
    if _java_common_internal._google_legacy_api_enabled():
        cc_info = cc_common.merge_cc_infos(
            cc_infos = [dep.cc_link_params_info for dep in runtime_deps + exports + deps] +
                       [cc_common.merge_cc_infos(cc_infos = native_libraries)],
        )
        result.update(
            cc_link_params_info = cc_info,
            transitive_native_libraries = cc_info.transitive_native_libraries(),
        )
    else:
        result.update(
            transitive_native_libraries = depset(
                transitive = [dep.transitive_native_libraries for dep in runtime_deps + exports + deps] +
                             [cc_common.merge_cc_infos(cc_infos = native_libraries).transitive_native_libraries()],
            ),
        )
    return result

JavaInfo, _new_javainfo = provider(
    doc = "Info object encapsulating all information by java rules.",
    fields = {
        "transitive_runtime_jars": "(depset[File]) A transitive set of jars required on the runtime classpath.",
        "transitive_compile_time_jars": "(depset[File]) The transitive set of jars required to build the target.",
        "compile_jars": """(depset[File]) The jars required directly at compile time. They can be interface jars
                (ijar or hjar), regular jars or both, depending on whether rule
                implementations chose to create interface jars or not.""",
        "full_compile_jars": """(depset[File]) The regular, full compile time Jars required by this target directly.
                They can be:
                 - the corresponding regular Jars of the interface Jars returned by JavaInfo.compile_jars
                 - the regular (full) Jars returned by JavaInfo.compile_jars

                Note: JavaInfo.compile_jars can return a mix of interface Jars and
                regular Jars.<p>Only use this method if interface Jars don't work with
                your rule set(s) (e.g. some Scala targets) If you're working with
                Java-only targets it's preferable to use interface Jars via
                JavaInfo.compile_jars""",
        "source_jars": """([File]) A list of Jars with all the source files (including those generated by
                annotations) of the target itself, i.e. NOT including the sources of the
                transitive dependencies.""",
        "outputs": "Deprecated: use java_outputs.",
        "annotation_processing": "Deprecated: Please use plugins instead.",
        "runtime_output_jars": "([File]) A list of runtime Jars created by this Java/Java-like target.",
        "transitive_deps": "Deprecated: Please use transitive_compile_time_jars instead.",
        "transitive_runtime_deps": "Deprecated: please use transitive_runtime_jars instead.",
        "transitive_source_jars": "(depset[File]) The Jars of all source files in the transitive closure.",
        "transitive_native_libraries": """(depset[LibraryToLink]) The transitive set of CC native
                libraries required by the target.""",
        "cc_link_params_info": "Deprecated. Do not use. C++ libraries to be linked into Java targets.",
        "module_flags_info": "(_ModuleFlagsInfo) The Java module flag configuration.",
        "plugins": """(_JavaPluginDataInfo) Data about all plugins that a consuming target should
               apply.
               This is typically either a `java_plugin` itself or a `java_library` exporting
               one or more plugins.
               A `java_library` runs annotation processing with all plugins from this field
               appearing in <code>deps</code> and `plugins` attributes.""",
        "api_generating_plugins": """"(_JavaPluginDataInfo) Data about API generating plugins
               defined or exported by this target.
               Those annotation processors are applied to a Java target before
               producing its header jars (which contain method signatures). When
               no API plugins are present, header jars are generated from the
               sources, reducing critical path.
               The `api_generating_plugins` is a subset of `plugins`.""",
        "java_outputs": "(_JavaOutputInfo) Information about outputs of this Java/Java-like target.",
        "compilation_info": """(java_compilation_info) Compilation information for this
               Java/Java-like target.""",
        "_transitive_full_compile_time_jars": "internal API, do not use",
        "_compile_time_java_dependencies": "internal API, do not use",
        "_neverlink": "internal API, do not use",
        "_plugin_info": "internal API, do not use",
    },
    init = _javainfo_init,
)
