// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Collection;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit Test for {@link ActionKeyCacher}. */
@RunWith(JUnit4.class)
public class ActionKeyCacherTest {
  private static final ArtifactExpander ARTIFACT_EXPANDER = artifact -> ImmutableSortedSet.of();

  private final ActionKeyCacher cacher;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  private static class TestActionKeyCacher extends ActionKeyCacher {
    @Override
    protected void computeKey(
        ActionKeyContext actionKeyContext,
        @Nullable ArtifactExpander artifactExpander,
        Fingerprint fp) {
      fp.addString(artifactExpander == null ? "no expander" : "has expander");
    }

    @Override
    public ActionOwner getOwner() {
      return null;
    }

    @Override
    public boolean isShareable() {
      return false;
    }

    @Override
    public String getMnemonic() {
      return null;
    }

    @Override
    public String prettyPrint() {
      return null;
    }

    @Override
    public String describe() {
      return null;
    }

    @Override
    public NestedSet<Artifact> getTools() {
      return null;
    }

    @Override
    public NestedSet<Artifact> getInputs() {
      return null;
    }

    @Override
    public NestedSet<Artifact> getSchedulingDependencies() {
      return null;
    }

    @Override
    public Collection<String> getClientEnvironmentVariables() {
      return null;
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      return null;
    }

    @Override
    public NestedSet<Artifact> getInputFilesForExtraAction(
        ActionExecutionContext actionExecutionContext) {
      return null;
    }

    @Override
    public ImmutableSet<Artifact> getMandatoryOutputs() {
      return null;
    }

    @Override
    public Artifact getPrimaryInput() {
      return null;
    }

    @Override
    public Artifact getPrimaryOutput() {
      return null;
    }

    @Override
    public NestedSet<Artifact> getMandatoryInputs() {
      return null;
    }

    @Override
    public MiddlemanType getActionType() {
      return null;
    }

    @Override
    public ImmutableMap<String, String> getExecProperties() {
      return ImmutableMap.of();
    }

    @Nullable
    @Override
    public PlatformInfo getExecutionPlatform() {
      return null;
    }
  }

  public ActionKeyCacherTest() {
    cacher = spy(new TestActionKeyCacher());
  }

  @Test
  public void getKey_withAndWithoutExpander_returnsDifferentKey() throws Exception {
    String withoutExpander = cacher.getKey(actionKeyContext, /*artifactExpander=*/ null);
    String withExpander = cacher.getKey(actionKeyContext, ARTIFACT_EXPANDER);

    assertThat(withExpander).isNotEqualTo(withoutExpander);
  }

  @Test
  public void getKey_withoutExpander_notStoredInCache() throws Exception {
    String key1 = cacher.getKey(actionKeyContext, /*artifactExpander=*/ null);
    verify(cacher).computeKey(eq(actionKeyContext), isNull(), any());

    String key2 = cacher.getKey(actionKeyContext, /*artifactExpander=*/ null);

    verify(cacher, times(2)).computeKey(eq(actionKeyContext), isNull(), any());
    assertThat(key1).isEqualTo(key2);
  }

  @Test
  public void getKey_withExpander_getsCacheHit() throws Exception {
    String key1 = cacher.getKey(actionKeyContext, ARTIFACT_EXPANDER);
    verify(cacher).computeKey(eq(actionKeyContext), eq(ARTIFACT_EXPANDER), any());

    String key2 = cacher.getKey(actionKeyContext, ARTIFACT_EXPANDER);

    verify(cacher).computeKey(eq(actionKeyContext), eq(ARTIFACT_EXPANDER), any());
    assertThat(key1).isEqualTo(key2);
  }

  @Test
  public void getKey_withoutExpander_skipsPrimedCache() throws Exception {
    String withExpander = cacher.getKey(actionKeyContext, ARTIFACT_EXPANDER);
    String withoutExpander = cacher.getKey(actionKeyContext, /* artifactExpander= */ null);

    verify(cacher).computeKey(eq(actionKeyContext), isNull(), any());
    assertThat(withExpander).isNotEqualTo(withoutExpander);
  }
}
