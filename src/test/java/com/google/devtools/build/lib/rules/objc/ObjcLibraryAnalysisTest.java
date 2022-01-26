// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.util.Pair;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code objc_library} that require performing analysis. */
@RunWith(JUnit4.class)
public final class ObjcLibraryAnalysisTest extends AnalysisTestCase {

  @Before
  public void setup() throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    useConfiguration(MockObjcSupport.requiredObjcCrosstoolFlags().toArray(new String[0]));
  }

  @Test
  public void libraryToLinkStaysInSyncWithConfiguredTarget() throws Exception {
    List<Pair<String, String>> builds =
        ImmutableList.of(
            Pair.of("clean build", "['a.m']"),
            Pair.of("action added", "['a.m', 'b.m']"),
            Pair.of("action removed", "['a.m']"));

    for (Pair<String, String> build : builds) {
      String context = build.getFirst();
      String srcs = build.getSecond();

      scratch.overwriteFile("foo/BUILD", "objc_library(name = 'lib', srcs = " + srcs + ")");
      update("//foo:lib");

      DerivedArtifact libraryToLink =
          (DerivedArtifact)
              getConfiguredTarget("//foo:lib")
                  .get(CcInfo.PROVIDER)
                  .getCcLinkingContext()
                  .getLibraries()
                  .getSingleton()
                  .getStaticLibrary();

      ActionLookupData generatingActionKey = libraryToLink.getGeneratingActionKey();
      ActionLookupValue actionLookupValue =
          (ActionLookupValue)
              skyframeExecutor
                  .getEvaluator()
                  .getExistingValue(generatingActionKey.getActionLookupKey());
      Action generatingAction = actionLookupValue.getAction(generatingActionKey.getActionIndex());

      assertWithMessage(context).that(generatingAction).isInstanceOf(CppLinkAction.class);
      assertWithMessage(context).that(generatingAction.getPrimaryOutput()).isEqualTo(libraryToLink);
    }
  }
}
