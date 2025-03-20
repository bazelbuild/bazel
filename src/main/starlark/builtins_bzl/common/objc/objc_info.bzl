# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""The provider ObjcInfo for ObjC rules. For more context see doc param of provider() call."""

def _objcinfo_init(
        j2objc_library = depset(),
        module_map = depset(),
        source = depset(),
        strict_include = depset(),
        umbrella_header = depset(),
        providers = []):  # List of depended-on ObjcInfo providers
    direct_module_maps = module_map.to_list()
    direct_sources = source.to_list()

    j2objc_library_transitive = []
    module_map_transitive = []
    source_transitive = []
    umbrella_header_transitive = []

    for p in providers:
        j2objc_library_transitive.append(p.j2objc_library)
        module_map_transitive.append(p.module_map)
        source_transitive.append(p.source)
        umbrella_header_transitive.append(p.umbrella_header)

    j2objc_library = depset(
        direct = j2objc_library.to_list(),
        transitive = j2objc_library_transitive,
    )
    module_map = depset(
        direct = direct_module_maps,
        transitive = module_map_transitive,
    )
    source = depset(
        direct = direct_sources,
        transitive = source_transitive,
    )
    umbrella_header = depset(
        direct = umbrella_header.to_list(),
        transitive = umbrella_header_transitive,
    )

    return {
        "direct_module_maps": direct_module_maps,
        "direct_sources": direct_sources,
        "j2objc_library": j2objc_library,
        "module_map": module_map,
        "source": source,
        "strict_include": strict_include,
        "umbrella_header": umbrella_header,
    }

ObjcInfo, _new_objcinfo = provider(
    """
A provider that provides all linking and miscellaneous information in the transitive closure of
its deps that are needed for building Objective-C rules. Most of the compilation information has
been migrated to {@code CcInfo}. The objc proto strict dependency include paths are still here
and stored in a field {@code strict_include} that is not propagated to any dependent ObjcInfo
provders and does not contain info from depended-on ObjcInfo providers.

The fields {@code direct_module_maps} and {@code direct_sources} only contain info from this target
whereas the fields {@code j2objc_library}, {@code module_map}, {@code source} and
{@code umbrella_header} are propagated to and from dependent and depended-on ObjcInfo providers
passed in via {@code providers} into the {@code _objcinfo_init} constructor.""",
    fields = {
        "direct_module_maps": """Module map files from this target directly
                (no transitive module maps).
                Used to enforce proper use of private header files and for Swift compilation.
                List<Artifact>""",
        "direct_sources": """All direct source files from this target (no transitive files),
                including any headers in the 'srcs' attribute. List<Artifact>""",
        "j2objc_library": """Static libraries that are built from J2ObjC-translated Java code.
                Depset<Artifact>""",
        "module_map": """Clang module maps, used to enforce proper use of private header files.
                Depset<Artifact>""",
        "source": "All transitive source files. Depset<Artifact>",
        "strict_include": """Non-propagated include search paths specified with '-I' on the
                command line. Also known as header search paths (and distinct from <em>user</em>
                header search paths). Depset<PathFragment>""",
        "umbrella_header": """Clang umbrella header. Public headers are #included in umbrella
                headers to be compatible with J2ObjC segmented headers. Depset<Artifact>""",
    },
    init = _objcinfo_init,
)
