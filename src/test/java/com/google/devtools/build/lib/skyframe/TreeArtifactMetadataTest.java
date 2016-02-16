// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.ActionInputHelper.asArtifactFiles;
import static org.junit.Assert.fail;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.cache.Digest;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Test the behavior of ActionMetadataHandler and ArtifactFunction
 * with respect to TreeArtifacts.
 */
@RunWith(JUnit4.class)
public class TreeArtifactMetadataTest extends ArtifactFunctionTestCase {

  // A list of subpaths for the SetArtifact created by our custom ActionExecutionFunction.
  private List<PathFragment> testTreeArtifactContents;

  @Before
  public final void setUp() throws Exception  {
    delegateActionExecutionFunction = new TreeArtifactExecutionFunction();
  }

  private TreeArtifactValue evaluateTreeArtifact(Artifact treeArtifact,
      Iterable<PathFragment> children)
    throws Exception {
    testTreeArtifactContents = ImmutableList.copyOf(children);
    for (PathFragment child : children) {
      file(treeArtifact.getPath().getRelative(child), child.toString());
    }
    return (TreeArtifactValue) evaluateArtifactValue(treeArtifact, /*mandatory=*/ true);
  }

  private TreeArtifactValue doTestTreeArtifacts(Iterable<PathFragment> children) throws Exception {
    Artifact output = createTreeArtifact("output");
    return doTestTreeArtifacts(output, children);
  }

  private TreeArtifactValue doTestTreeArtifacts(Artifact tree,
      Iterable<PathFragment> children)
      throws Exception {
    TreeArtifactValue value = evaluateTreeArtifact(tree, children);
    assertThat(value.getChildPaths()).containsExactlyElementsIn(ImmutableSet.copyOf(children));
    assertThat(value.getChildren(tree)).containsExactlyElementsIn(
        asArtifactFiles(tree, children));

    // Assertions about digest. As of this writing this logic is essentially the same
    // as that in TreeArtifact, but it's good practice to unit test anyway to guard against
    // breaking changes.
    Map<String, Metadata> digestBuilder = new HashMap<>();
    for (PathFragment child : children) {
      Metadata subdigest = new Metadata(tree.getPath().getRelative(child).getMD5Digest());
      digestBuilder.put(child.getPathString(), subdigest);
    }
    assertThat(Digest.fromMetadata(digestBuilder).asMetadata().digest).isEqualTo(value.getDigest());
    return value;
  }

  @Test
  public void testEmptyTreeArtifacts() throws Exception {
    TreeArtifactValue value = doTestTreeArtifacts(ImmutableList.<PathFragment>of());
    // Additional test, only for this test method: we expect the Metadata is equal to
    // the digest [0, 0, ...]
    assertThat(value.getMetadata().digest).isEqualTo(value.getDigest());
    // Java zero-fills arrays.
    assertThat(value.getDigest()).isEqualTo(new byte[16]);
  }

  @Test
  public void testTreeArtifactsWithDigests() throws Exception {
    fastDigest = true;
    doTestTreeArtifacts(ImmutableList.of(new PathFragment("one")));
  }

  @Test
  public void testTreeArtifactsWithoutDigests() throws Exception {
    fastDigest = false;
    doTestTreeArtifacts(ImmutableList.of(new PathFragment("one")));
  }

  @Test
  public void testTreeArtifactMultipleDigests() throws Exception {
    doTestTreeArtifacts(ImmutableList.of(new PathFragment("one"), new PathFragment("two")));
  }

  @Test
  public void testIdenticalTreeArtifactsProduceTheSameDigests() throws Exception {
    // Make sure different root dirs for set artifacts don't produce different digests.
    Artifact one = createTreeArtifact("outOne");
    Artifact two = createTreeArtifact("outTwo");
    ImmutableList<PathFragment> children =
        ImmutableList.of(new PathFragment("one"), new PathFragment("two"));
    TreeArtifactValue valueOne = evaluateTreeArtifact(one, children);
    TreeArtifactValue valueTwo = evaluateTreeArtifact(two, children);
    assertThat(valueOne.getDigest()).isEqualTo(valueTwo.getDigest());
  }

  /**
   * Tests that ArtifactFunction rethrows transitive {@link IOException}s as
   * {@link MissingInputFileException}s.
   */
  @Test
  public void testIOExceptionEndToEnd() throws Throwable {
    final IOException exception = new IOException("boop");
    setupRoot(
        new CustomInMemoryFs() {
          @Override
          public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
            if (path.getBaseName().equals("one")) {
              throw exception;
            }
            return super.stat(path, followSymlinks);
          }
        });
    try {
      Artifact artifact = createTreeArtifact("outOne");
      TreeArtifactValue value = evaluateTreeArtifact(artifact,
          ImmutableList.of(new PathFragment("one")));
      fail("MissingInputFileException expected, got " + value);
    } catch (Exception e) {
      assertThat(Throwables.getRootCause(e).getMessage()).contains(exception.getMessage());
    }
  }

  private void file(Path path, String contents) throws Exception {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    writeFile(path, contents);
  }

  private Artifact createTreeArtifact(String path) throws IOException {
    PathFragment execPath = new PathFragment("out").getRelative(path);
    Path fullPath = root.getRelative(execPath);
    Artifact output =
        new SpecialArtifact(
            fullPath, Root.asDerivedRoot(root, root.getRelative("out")), execPath, ALL_OWNER,
            SpecialArtifactType.TREE);
    actions.add(new DummyAction(ImmutableList.<Artifact>of(), output));
    FileSystemUtils.createDirectoryAndParents(fullPath);
    return output;
  }

  private ArtifactValue evaluateArtifactValue(Artifact artifact, boolean mandatory)
      throws Exception {
    SkyKey key = ArtifactValue.key(artifact, mandatory);
    EvaluationResult<ArtifactValue> result = evaluate(key);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    return result.get(key);
  }

  private void setGeneratingActions() {
    if (evaluator.getExistingValueForTesting(OWNER_KEY) == null) {
      differencer.inject(ImmutableMap.of(OWNER_KEY, new ActionLookupValue(actions)));
    }
  }

  private <E extends SkyValue> EvaluationResult<E> evaluate(SkyKey... keys)
      throws InterruptedException {
    setGeneratingActions();
    return driver.evaluate(
        Arrays.asList(keys), /*keepGoing=*/
        false,
        SkyframeExecutor.DEFAULT_THREAD_COUNT,
        NullEventHandler.INSTANCE);
  }

  private class TreeArtifactExecutionFunction implements SkyFunction {
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
      Map<ArtifactFile, FileValue> fileData = new HashMap<>();
      Map<PathFragment, FileArtifactValue> treeArtifactData = new HashMap<>();
      Action action = (Action) skyKey.argument();
      Artifact output = Iterables.getOnlyElement(action.getOutputs());
      for (PathFragment subpath : testTreeArtifactContents) {
        try {
          ArtifactFile suboutput = ActionInputHelper.artifactFile(output, subpath);
          FileValue fileValue = ActionMetadataHandler.fileValueFromArtifactFile(
              suboutput, null, tsgm);
          fileData.put(suboutput, fileValue);
          // Ignore FileValue digests--correctness of these digests is not part of this tests.
          byte[] digest = DigestUtils.getDigestOrFail(suboutput.getPath(), 1);
          treeArtifactData.put(suboutput.getParentRelativePath(),
              FileArtifactValue.createWithDigest(suboutput.getPath(), digest, fileValue.getSize()));
        } catch (IOException e) {
          throw new SkyFunctionException(e, Transience.TRANSIENT) {};
        }
      }

      TreeArtifactValue treeArtifactValue = TreeArtifactValue.create(treeArtifactData);

      return new ActionExecutionValue(
          fileData,
          ImmutableMap.of(output, treeArtifactValue),
          ImmutableMap.<Artifact, FileArtifactValue>of());
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }
}
