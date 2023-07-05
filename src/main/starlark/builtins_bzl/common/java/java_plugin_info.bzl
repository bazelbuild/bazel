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
Definition of JavaPluginInfo provider.
"""

load(":common/java/java_common_internal_for_builtins.bzl", _merge_private_for_builtins = "merge")

_JavaPluginDataInfo = provider(
    doc = "Provider encapsulating information about a Java compatible plugin.",
    fields = {
        "processor_classes": "depset(str) The fully qualified classnames of entry points for the compiler",
        "processor_jars": "depset(file) Deps containing an annotation processor",
        "processor_data": "depset(file) Files needed during execution",
    },
)

EMPTY_PLUGIN_DATA = _JavaPluginDataInfo(
    processor_classes = depset(),
    processor_jars = depset(),
    processor_data = depset(),
)

def _javaplugininfo_init(
        runtime_deps,
        processor_class,
        data = [],
        generates_api = False):
    """ Constructs JavaPluginInfo

    Args:
        runtime_deps: ([JavaInfo]) list of deps containing an annotation
             processor.
        processor_class: (String) The fully qualified class name that the Java
             compiler uses as an entry point to the annotation processor.
        data: (depset[File]) The files needed by this annotation
             processor during execution.
        generates_api: (boolean) Set to true when this annotation processor
            generates API code. Such an annotation processor is applied to a
            Java target before producing its header jars (which contains method
            signatures). When no API plugins are present, header jars are
            generated from the sources, reducing the critical path.
            WARNING: This parameter affects build performance, use it only if
            necessary.

    Returns:
        (JavaPluginInfo)
    """

    # we don't need the private API but java_common needs JavaPluginInfo which would be a cycle
    java_infos = _merge_private_for_builtins(runtime_deps)
    processor_data = data if type(data) == "depset" else depset(data)
    plugins = _JavaPluginDataInfo(
        processor_classes = depset([processor_class]) if processor_class else depset(),
        processor_jars = java_infos.transitive_runtime_jars,
        processor_data = processor_data,
    )
    return {
        "plugins": plugins,
        "api_generating_plugins": plugins if generates_api else EMPTY_PLUGIN_DATA,
        "java_outputs": java_infos.java_outputs,
    }

JavaPluginInfo, _new_javaplugininfo = provider(
    doc = "Provider encapsulating information about Java plugins.",
    fields = {
        "plugins": """
            Returns data about all plugins that a consuming target should apply.
            This is typically either a <code>java_plugin</code> itself or a
            <code>java_library</code> exporting one or more plugins.
            A <code>java_library</code> runs annotation processing with all
            plugins from this field appearing in <code>deps</code> and
            <code>plugins</code> attributes.""",
        "api_generating_plugins": """
            Returns data about API generating plugins defined or exported by
            this target.
            Those annotation processors are applied to a Java target before
            producing its header jars (which contain method signatures). When
            no API plugins are present, header jars are generated from the
            sources, reducing critical path.
            The <code>api_generating_plugins</code> is a subset of
            <code>plugins</code>.""",
        "java_outputs": """
            Returns information about outputs of this Java/Java-like target.
        """,
    },
    init = _javaplugininfo_init,
)

def merge_without_outputs(infos):
    """ Merge plugin information from a list of JavaPluginInfo or JavaInfo

    Args:
        infos: ([JavaPluginInfo|JavaInfo]) list of providers to merge

    Returns:
        (JavaPluginInfo)
    """
    plugins = []
    api_generating_plugins = []
    for info in infos:
        if _has_plugin_data(info.plugins):
            plugins.append(info.plugins)
        if _has_plugin_data(info.api_generating_plugins):
            api_generating_plugins.append(info.api_generating_plugins)
    return _new_javaplugininfo(
        plugins = _merge_plugin_data(plugins),
        api_generating_plugins = _merge_plugin_data(api_generating_plugins),
        java_outputs = [],
    )

def _has_plugin_data(plugin_data):
    return plugin_data and (
        plugin_data.processor_classes or
        plugin_data.processor_jars or
        plugin_data.processor_data
    )

def _merge_plugin_data(datas):
    return _JavaPluginDataInfo(
        processor_classes = depset(transitive = [p.processor_classes for p in datas]),
        processor_jars = depset(transitive = [p.processor_jars for p in datas]),
        processor_data = depset(transitive = [p.processor_data for p in datas]),
    )
