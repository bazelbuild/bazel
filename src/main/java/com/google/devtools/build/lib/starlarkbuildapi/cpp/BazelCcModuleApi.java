// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.StarlarkBuiltin;

/** Utilites related to C++ support. */
@StarlarkBuiltin(
    name = "cc_common",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc = "Utilities for C++ compilation, linking, and command line generation.")
public interface BazelCcModuleApi<
        StarlarkActionFactoryT extends StarlarkActionFactoryApi,
        FileT extends FileApi,
        FdoContextT extends FdoContextApi<?>,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcToolchainProviderT extends
            CcToolchainProviderApi<
                    FeatureConfigurationT,
                    ?,
                    FdoContextT,
                    ConstraintValueT,
                    StarlarkRuleContextT,
                    ?,
                    ? extends CppConfigurationApi<?>,
                    CcToolchainVariablesT>,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CompilationContextT extends CcCompilationContextApi<FileT>,
        CompilationOutputsT extends CcCompilationOutputsApi<FileT>,
        LinkingOutputsT extends CcLinkingOutputsApi<FileT, LtoBackendArtifactsT>,
        LtoBackendArtifactsT extends LtoBackendArtifactsApi<FileT>,
        LinkerInputT extends LinkerInputApi<LibraryToLinkT, LtoBackendArtifactsT, FileT>,
        LibraryToLinkT extends LibraryToLinkApi<FileT, LtoBackendArtifactsT>,
        LinkingContextT extends CcLinkingContextApi<FileT>,
        CcToolchainVariablesT extends CcToolchainVariablesApi,
        CcToolchainConfigInfoT extends CcToolchainConfigInfoApi,
        DebugContextT extends CcDebugInfoContextApi,
        CppModuleMapT extends CppModuleMapApi<FileT>>
    extends CcModuleApi<
        StarlarkActionFactoryT,
        FileT,
        FdoContextT,
        CcToolchainProviderT,
        FeatureConfigurationT,
        CompilationContextT,
        LtoBackendArtifactsT,
        LinkerInputT,
        LinkingContextT,
        LibraryToLinkT,
        CcToolchainVariablesT,
        ConstraintValueT,
        StarlarkRuleContextT,
        CcToolchainConfigInfoT,
        CompilationOutputsT,
        DebugContextT,
        CppModuleMapT,
        LinkingOutputsT> {}
