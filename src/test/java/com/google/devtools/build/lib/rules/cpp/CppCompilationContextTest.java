// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link CppCompilationContext}.
 */
@RunWith(JUnit4.class)
public class CppCompilationContextTest extends BuildViewTestCase {

  @Before
  public final void createBuildFile() throws Exception {
    scratch.file("foo/BUILD",
        "cc_binary(name = 'foo',",
        "          srcs = ['foo.cc'])",
        "cc_binary(name = 'bar',",
        "          srcs = ['bar.cc'])");
  }

  @Test
  public void testEqualsAndHashCode() throws Exception {
    MiddlemanFactory middlemanFactory = getTestAnalysisEnvironment().getMiddlemanFactory();
    ConfiguredTarget fooBin = getConfiguredTarget("//foo:foo");
    CppCompilationContext fooContextA1 = new CppCompilationContext.Builder(
        getRuleContext(fooBin)).build(NULL_ACTION_OWNER, middlemanFactory);
    CppCompilationContext fooContextA2 = new CppCompilationContext.Builder(
        getRuleContext(fooBin)).build(NULL_ACTION_OWNER, middlemanFactory);
    CppCompilationContext fooContextB = new CppCompilationContext.Builder(
        getRuleContext(fooBin)).addDefine("a=b")
            .build(NULL_ACTION_OWNER, middlemanFactory);
    ConfiguredTarget barBin = getConfiguredTarget("//foo:bar");
    // The fact that the configured target is different does not matter in this case.
    CppCompilationContext barContext = new CppCompilationContext.Builder(
        getRuleContext(barBin)).build(NULL_ACTION_OWNER, middlemanFactory);

    CppCompilationContext fooContextWithModuleMap =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .setCppModuleMap(new CppModuleMap(getBinArtifact("foo/something.xyz", fooBin), "name"))
        .build(NULL_ACTION_OWNER, middlemanFactory);

    CppCompilationContext fooContextWithCompilationPrerequisites =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .addCompilationPrerequisites(ImmutableList.of(getBinArtifact("foo/something.xyz", fooBin)))
        .build(NULL_ACTION_OWNER, middlemanFactory);

    CppCompilationContext fooContextWithModuleMapAndCompilationPrerequisites =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .addCompilationPrerequisites(ImmutableList.of(getBinArtifact("foo/something.xyz", fooBin)))
        .setCppModuleMap(new CppModuleMap(getBinArtifact("foo/something.xyz", fooBin), "name"))
        .build(NULL_ACTION_OWNER, middlemanFactory);

    CppCompilationContext fooContextWithInheritedModuleMapSameAsModuleMap =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .mergeDependentContext(fooContextWithModuleMap)
        .setCppModuleMap(new CppModuleMap(getBinArtifact("foo/something.xyz", fooBin), "name"))
        .build(NULL_ACTION_OWNER, middlemanFactory);

    CppCompilationContext fooContextWithHeaderModuleSrcs =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .addDeclaredIncludeSrc(getSourceArtifact("foo/foo.h"))
        .build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextWithInheritedHeaderModuleSrcs =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .mergeDependentContext(fooContextWithHeaderModuleSrcs)
        .build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextWithHeaderModule =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .setHeaderModule(getGenfilesArtifact("foo/something.pcm", fooBin))
        .addDeclaredIncludeSrc(getSourceArtifact("foo/bar.h"))
        .build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextWithPicHeaderModule =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .setPicHeaderModule(getGenfilesArtifact("foo/something.pcm", fooBin))
        .build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextWithInheritedHeaderModule =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .mergeDependentContext(fooContextWithHeaderModule)
        .setHeaderModule(getGenfilesArtifact("foo/something.pcm", fooBin))
        .addDeclaredIncludeSrc(getSourceArtifact("foo/bar.h"))
        .build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextWithTransitivelyInheritedHeaderModule =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
        .mergeDependentContext(fooContextWithInheritedHeaderModule)
        .setHeaderModule(getGenfilesArtifact("foo/something.pcm", fooBin))
        .build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextUsingHeaderModules =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
            .setProvideTransitiveModuleMaps(true).build(NULL_ACTION_OWNER, middlemanFactory);
    
    CppCompilationContext fooContextNotUsingHeaderModules =
        new CppCompilationContext.Builder(getRuleContext(fooBin))
            .setUseHeaderModules(true)
            .build(NULL_ACTION_OWNER, middlemanFactory);

    new EqualsTester()
        .addEqualityGroup(fooContextA1, fooContextA2, barContext)
        .addEqualityGroup(fooContextB)
        .addEqualityGroup(fooContextWithModuleMap)
        .addEqualityGroup(fooContextWithCompilationPrerequisites)
        .addEqualityGroup(fooContextWithModuleMapAndCompilationPrerequisites)
        .addEqualityGroup(fooContextWithInheritedModuleMapSameAsModuleMap)
        .addEqualityGroup(fooContextWithHeaderModuleSrcs)
        .addEqualityGroup(fooContextWithInheritedHeaderModuleSrcs)
        .addEqualityGroup(fooContextWithHeaderModule)
        .addEqualityGroup(fooContextWithPicHeaderModule)
        .addEqualityGroup(fooContextWithInheritedHeaderModule)
        .addEqualityGroup(fooContextWithTransitivelyInheritedHeaderModule)
        .addEqualityGroup(fooContextUsingHeaderModules)
        .addEqualityGroup(fooContextNotUsingHeaderModules)
        .testEquals();
  }
}
