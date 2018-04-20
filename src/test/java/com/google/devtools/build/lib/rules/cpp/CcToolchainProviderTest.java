// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoMode;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@code CcToolchainProvider}
 */
@RunWith(JUnit4.class)
public class CcToolchainProviderTest {
  @Test
  public void equalityIsObjectIdentity() throws Exception {
    CcToolchainProvider a =
        new CcToolchainProvider(
            /* values= */ ImmutableMap.of(),
            /* cppConfiguration= */ null,
            /* toolchainInfo= */ null,
            /* crosstoolTopPathFragment= */ null,
            /* crosstool= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* crosstoolMiddleman= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* compile= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* strip= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* objCopy= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* as= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* ar= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* link= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* interfaceSoBuilder= */ null,
            /* dwp= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* coverage= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* libcLink= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* dynamicRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeSolibDir= */ PathFragment.EMPTY_FRAGMENT,
            CcCompilationContextInfo.EMPTY,
            /* supportsParamFiles= */ false,
            /* supportsHeaderParsing= */ false,
            Variables.EMPTY,
            /* builtinIncludeFiles= */ ImmutableList.<Artifact>of(),
            /* coverageEnvironment= */ NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
            /* linkDynamicLibraryTool= */ null,
            /* builtInIncludeDirectories= */ ImmutableList.<PathFragment>of(),
            /* sysroot= */ null,
            FdoMode.OFF,
            /* useLLVMCoverageMapFormat= */ false,
            /* codeCoverageEnabled= */ false,
            /* isHostConfiguration= */ false);

    CcToolchainProvider b =
        new CcToolchainProvider(
            /* values= */ ImmutableMap.of(),
            /* cppConfiguration= */ null,
            /* toolchainInfo= */ null,
            /* crosstoolTopPathFragment= */ null,
            /* crosstool= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* crosstoolMiddleman= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* compile= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* strip= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* objCopy= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* as= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* ar= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* link= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* interfaceSoBuilder= */ null,
            /* dwp= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* coverage= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* libcLink= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* dynamicRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeSolibDir= */ PathFragment.EMPTY_FRAGMENT,
            CcCompilationContextInfo.EMPTY,
            /* supportsParamFiles= */ false,
            /* supportsHeaderParsing= */ false,
            Variables.EMPTY,
            /* builtinIncludeFiles= */ ImmutableList.<Artifact>of(),
            /* coverageEnvironment= */ NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
            /* linkDynamicLibraryTool= */ null,
            /* builtInIncludeDirectories= */ ImmutableList.<PathFragment>of(),
            /* sysroot= */ null,
            FdoMode.OFF,
            /* useLLVMCoverageMapFormat= */ false,
            /* codeCoverageEnabled= */ false,
            /* isHostConfiguration= */ false);

    new EqualsTester()
        .addEqualityGroup(a)
        .addEqualityGroup(b)
        .testEquals();
  }
}
