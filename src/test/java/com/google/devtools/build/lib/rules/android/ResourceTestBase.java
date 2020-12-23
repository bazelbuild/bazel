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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsValue;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyFunction;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.After;
import org.junit.Before;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;

/** Base class for tests that work with resource artifacts. */
public abstract class ResourceTestBase extends AndroidBuildViewTestCase {
  public static final String RESOURCE_ROOT = "java/android/res";

  private static final ImmutableSet<String> TOOL_FILENAMES =
      ImmutableSet.of(
          "static_aapt_tool",
          "aapt.static",
          "aapt",
          "static_aapt2_tool",
          "aapt2",
          "empty.sh",
          "android_blaze.jar",
          "android.jar",
          "ResourceProcessorBusyBox_deploy.jar");

  private static final ArtifactOwner OWNER =
      () -> {
        try {
          return Label.create("java", "all");
        } catch (LabelSyntaxException e) {
          assertWithMessage(e.getMessage()).fail();
          return null;
        }
      };

  /** A faked {@link RuleErrorConsumer} that validates that only expected errors were reported. */
  @RunWith(Enclosed.class)
  public static final class FakeRuleErrorConsumer implements RuleErrorConsumer {
    private String ruleErrorMessage = null;
    private String attributeErrorAttribute = null;
    private String attributeErrorMessage = null;

    private final List<String> ruleWarnings = new ArrayList<>();

    // Use an ArrayListMultimap since it allows duplicates - we'll want to know if a warning is
    // reported twice.
    private final Multimap<String, String> attributeWarnings = ArrayListMultimap.create();

    @Override
    public void ruleWarning(String message) {
      ruleWarnings.add(message);
    }

    @Override
    public void ruleError(String message) {
      ruleErrorMessage = message;
    }

    @Override
    public void attributeWarning(String attrName, String message) {
      attributeWarnings.put(attrName, message);
    }

    @Override
    public void attributeError(String attrName, String message) {
      attributeErrorAttribute = attrName;
      attributeErrorMessage = message;
    }

    @Override
    public boolean hasErrors() {
      return ruleErrorMessage != null || attributeErrorMessage != null;
    }

    public Collection<String> getAndClearRuleWarnings() {
      Collection<String> warnings = ImmutableList.copyOf(ruleWarnings);
      ruleWarnings.clear();
      return warnings;
    }

    public void assertNoRuleWarnings() {
      assertThat(ruleWarnings).isEmpty();
    }

    public Collection<String> getAndClearAttributeWarnings(String attrName) {
      if (!attributeWarnings.containsKey(attrName)) {
        return ImmutableList.of();
      }

      return attributeWarnings.removeAll(attrName);
    }

    public void assertNoAttributeWarnings(String attrName) {
      assertThat(attributeWarnings).doesNotContainKey(attrName);
    }

    /**
     * Called at the end of a test to assert that that test produced a rule error
     *
     * @param expectedMessage a substring of the expected message
     */
    public void assertRuleError(String expectedMessage) {
      // Clear the message before asserting so that if we fail here the error is not masked by the
      // @After call to assertNoUnexpectedErrors.

      String message = ruleErrorMessage;
      ruleErrorMessage = null;

      assertThat(message).contains(expectedMessage);
    }

    /**
     * Called at the end of a test to assert that that test produced an attribute error
     *
     * @param expectedAttribute the attribute that caused the error
     * @param expectedMessage a substring of the expected message
     */
    public void assertAttributeError(String expectedAttribute, String expectedMessage) {
      // Clear the message before asserting so that if we fail here the error is not masked by the
      // @After call to assertNoUnexpectedErrors.
      String attr = attributeErrorAttribute;
      String message = attributeErrorMessage;
      attributeErrorAttribute = null;
      attributeErrorMessage = null;

      assertThat(message).contains(expectedMessage);
      assertThat(attr).isEqualTo(expectedAttribute);
    }

    /**
     * Asserts this {@link RuleErrorConsumer} encountered no unexpected errors. To consume an
     * expected error, call {@link #assertRuleError(String)} or {@link #assertAttributeError(String,
     * String)} in your test after the error is produced.
     */
    private void assertNoUnexpectedErrors() {
      assertThat(ruleErrorMessage).isNull();
      assertThat(attributeErrorMessage).isNull();
      assertThat(attributeErrorAttribute).isNull();
    }
  }

  public FakeRuleErrorConsumer errorConsumer;
  public FileSystem fileSystem;
  public ArtifactRoot root;

  @Before
  public void setup() throws Exception {
    errorConsumer = new FakeRuleErrorConsumer();
    fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    root = ArtifactRoot.asSourceRoot(Root.fromPath(fileSystem.getPath("/")));
  }

  @After
  public void assertNoErrors() {
    errorConsumer.assertNoUnexpectedErrors();
  }

  public ImmutableList<Artifact> getResources(String... pathStrings) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (String pathString : pathStrings) {
      builder.add(getResource(pathString));
    }

    return builder.build();
  }

  Artifact getResource(String pathString) {
    return getArtifact(RESOURCE_ROOT, pathString);
  }

  Artifact getOutput(String pathString) {
    return getArtifact("outputs", pathString);
  }

  private Artifact getArtifact(String subdir, String pathString) {
    Path path = fileSystem.getPath("/" + subdir + "/" + pathString);
    return new Artifact.SourceArtifact(
        root, root.getExecPath().getRelative(root.getRoot().relativize(path)), OWNER);
  }

  /**
   * Gets a RuleContext that can be used to register actions and test that they are created
   * correctly.
   *
   * <p>Takes in a dummy target which will be used to configure the RuleContext's {@link
   * AndroidConfiguration}.
   */
  public RuleContext getRuleContextForActionTesting(ConfiguredTarget dummyTarget) throws Exception {
    RuleContext dummy = getRuleContext(dummyTarget);
    ExtendedEventHandler eventHandler = new StoredEventHandler();

    SkyFunction.Environment skyframeEnv =
        skyframeExecutor.getSkyFunctionEnvironmentForTesting(eventHandler);
    StarlarkBuiltinsValue starlarkBuiltinsValue =
        (StarlarkBuiltinsValue)
            Preconditions.checkNotNull(skyframeEnv.getValue(StarlarkBuiltinsValue.key()));
    CachingAnalysisEnvironment analysisEnv =
        new CachingAnalysisEnvironment(
            view.getArtifactFactory(),
            skyframeExecutor.getActionKeyContext(),
            ConfiguredTargetKey.builder()
                .setLabel(dummyTarget.getLabel())
                .setConfiguration(targetConfig)
                .build(),
            /*isSystemEnv=*/ false,
            targetConfig.extendedSanityChecks(),
            targetConfig.allowAnalysisFailures(),
            eventHandler,
            skyframeEnv,
            starlarkBuiltinsValue);

    return view.getRuleContextForTesting(
        eventHandler,
        dummyTarget,
        analysisEnv,
        new BuildConfigurationCollection(
            ImmutableList.of(dummy.getConfiguration()), dummy.getHostConfiguration()));
  }

  /**
   * Assets that the action used to generate the given outputs has the expected inputs and outputs.
   */
  void assertActionArtifacts(
      RuleContext ruleContext, ImmutableList<Artifact> inputs, ImmutableList<Artifact> outputs) {
    // Actions must have at least one output
    assertThat(outputs).isNotEmpty();

    // Get the action from one of the outputs
    ActionAnalysisMetadata action =
        ruleContext.getAnalysisEnvironment().getLocalGeneratingAction(outputs.get(0));
    assertThat(action).isNotNull();

    assertThat(removeToolingArtifacts(action.getInputs())).containsExactlyElementsIn(inputs);

    assertThat(action.getOutputs()).containsExactlyElementsIn(outputs);
  }

  /** Remove busybox and aapt2 tooling artifacts from a list of action inputs */
  private static Iterable<Artifact> removeToolingArtifacts(NestedSet<Artifact> inputArtifacts) {
    return inputArtifacts.toList().stream()
        .filter(
            artifact ->
                // Not a known tool
                !TOOL_FILENAMES.contains(artifact.getFilename())
                    // Not one of the various busybox tools (we get different ones on different OSs)
                    && !artifact.getFilename().contains("busybox")
                    // Not a params file
                    && !artifact.getFilename().endsWith(".params"))
        .collect(Collectors.toList());
  }
}
