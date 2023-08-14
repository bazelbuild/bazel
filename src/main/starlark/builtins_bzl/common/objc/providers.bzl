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

"""Definition of j2objc providers."""

J2ObjcMappingFileInfo = provider(
    doc = "Provider which contains mapping files to export mappings required by J2ObjC translation and proto compilation.",
    fields = dict(
        header_mapping_files = "(depset[File]) Files which map Java classes to their associated translated ObjC header. Used by J2ObjC to output correct import directive during translation.",
        class_mapping_files = "(depset[File]) Files which map Java class names to their associated ObjC class names. Used to support J2ObjC package prefixes.",
        dependency_mapping_files = "(depset[File]) Files which map translated ObjC files to their translated direct dependency files. Used to support J2ObjC dead code analysis and removal.",
        archive_source_mapping_files = "(depset[File]) Files containing mappings between J2ObjC static library archives and their associated J2ObjC-translated source files.",
    ),
)

J2ObjcEntryClassInfo = provider(
    doc = "Provider which is exported by j2objc_library to export entry class information necessary for J2ObjC dead code removal performed at the binary level in ObjC rules.",
    fields = dict(
        entry_classes = "(depset[str]) Depset of entry classes specified on attribute entry_classes of j2objc_library targets transitively.",
    ),
)
