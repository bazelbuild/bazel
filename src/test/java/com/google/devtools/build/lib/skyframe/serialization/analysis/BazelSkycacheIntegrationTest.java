// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.devtools.build.lib.cmdline.Label.parseCanonicalUnchecked;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutorOwner;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommandStartEvent;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.WorkspaceStatusValue;
import com.google.devtools.build.lib.skyframe.serialization.DeserializedSkyValue;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.InMemoryFingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.KeyBytesProvider;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SerializationModule;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatus;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses;
import com.google.devtools.build.lib.skyframe.serialization.proto.DataType;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class BazelSkycacheIntegrationTest extends SkycacheIntegrationTestBase {
  private final LongVersionGetter versionGetter = mock(LongVersionGetter.class);
  private final FailingFingerprintValueStore failingStore = new FailingFingerprintValueStore();

  @Before
  public void injectVersionGetter() {
    runtimeWrapper.registerSubscriber(
        new Object() {
          @Subscribe
          public void commandStart(CommandStartEvent event) {
            CommandEnvironment env = getCommandEnvironment();
            env.setVersionGetter(versionGetter);
          }
        });
  }

  private static class FailingFingerprintValueStore implements FingerprintValueStore {
    private final InMemoryFingerprintValueStore delegate = new InMemoryFingerprintValueStore();
    private final AtomicBoolean shouldFail = new AtomicBoolean();
    private final AtomicInteger failCounter = new AtomicInteger();
    private final AtomicReference<KeyBytesProvider> lastFailedKey = new AtomicReference<>();

    private void failNextPut() {
      shouldFail.set(true);
    }

    private int getFailCounter() {
      return failCounter.get();
    }

    private KeyBytesProvider getFailedKey() {
      return lastFailedKey.get();
    }

    @Override
    public WriteStatus put(KeyBytesProvider fingerprint, byte[] serializedBytes) {
      if (shouldFail.getAndSet(false)) {
        failCounter.getAndIncrement();
        lastFailedKey.set(fingerprint);
        return WriteStatuses.immediateFailedWriteStatus(
            new IOException("Simulated write failure for " + fingerprint));
      }
      return delegate.put(fingerprint, serializedBytes);
    }

    @Override
    public ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) throws IOException {
      return delegate.get(fingerprint);
    }
  }

  private class ModuleWithOverrides extends SerializationModule {
    @Override
    protected RemoteAnalysisCachingServicesSupplier getAnalysisCachingServicesSupplier(
        BlazeRuntime runtime) {
      return new TestServicesSupplier(failingStore);
    }
  }

  private static class TestServicesSupplier implements RemoteAnalysisCachingServicesSupplier {
    private final ListenableFuture<FingerprintValueStore> fingerprintValueStore;
    private final RemoteAnalysisCacheClient client = mock(RemoteAnalysisCacheClient.class);

    /** This is safe and does not leak because the underlying commonPool is never shut down. */
    private final SafeExecutorOwner commandExecutor = new SafeExecutorOwner(commonPool());

    private TestServicesSupplier(FailingFingerprintValueStore failingStore) {
      this.fingerprintValueStore = immediateFuture(failingStore);
      when(client.getStats()).thenReturn(RemoteAnalysisCacheClient.EMPTY_STATS);
      when(client.lookup(any()))
          .thenAnswer(
              invocation -> {
                byte[] key = invocation.getArgument(0);
                ByteString entry =
                    failingStore.delegate.fingerprintToContents.get(ByteString.copyFrom(key));
                if (entry == null) {
                  return immediateFuture(new LookupResultImpl(new byte[0]));
                }
                CodedInputStream input = entry.newCodedInput();
                byte[] invalidationFingerprint =
                    switch (DataType.forNumber(input.readEnum())) {
                      case DATA_TYPE_EMPTY -> null;
                      case DATA_TYPE_FILE, DATA_TYPE_LISTING -> {
                        input.readString();
                        yield null;
                      }
                      case DATA_TYPE_ANALYSIS_NODE, DATA_TYPE_EXECUTION_NODE ->
                          PackedFingerprint.readFrom(input).toBytes();
                      default -> throw new IOException("Unexpected analysis cache entry header");
                    };
                return immediateFuture(
                    new LookupResultImpl(
                        Arrays.copyOfRange(
                            entry.toByteArray(), input.getTotalBytesRead(), entry.size()),
                        invalidationFingerprint,
                        0));
              });
    }

    @Override
    public ListenableFuture<FingerprintValueStore> getFingerprintValueStore() {
      return fingerprintValueStore;
    }

    @Override
    public ListenableFuture<RemoteAnalysisCacheClient> getAnalysisCacheClient() {
      return immediateFuture(client);
    }

    @Override
    public SafeExecutor getCommandExecutor() {
      return commandExecutor;
    }

    @Override
    public void resetCommandState() {}
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new ModuleWithOverrides())
        .addBlazeService(new CompressionServiceImpl());
  }

  @Test
  public void cqueryAfterCacheHitPreservesUnrelatedDirtyPackage() throws Exception {
    assertConfiguredQueryAfterCacheHitPreservesUnrelatedDirtyPackage(false);
  }

  @Test
  public void aqueryAfterCacheHitPreservesUnrelatedDirtyPackage() throws Exception {
    assertConfiguredQueryAfterCacheHitPreservesUnrelatedDirtyPackage(true);
  }

  private void assertConfiguredQueryAfterCacheHitPreservesUnrelatedDirtyPackage(boolean aquery)
      throws Exception {
    write(
        "scope/BUILD",
        "genrule(name = 'subject', outs = ['out'], cmd = 'touch $@')",
        "filegroup(name = 'both', srcs = [':subject', '//other:local'])");
    write("other/BUILD", "exports_files(['local'])");
    write("other/local", "contents");
    addOptions(
        "--experimental_active_directories=other",
        "--nobuild",
        "--version_window_for_dirty_node_gc=-1");
    assertUploadSuccess("//scope:subject");
    getSkyframeExecutor().resetEvaluator();
    addOptions(OFF_MODE_OPTION);
    buildTarget("//other:local");
    var packageKey = parseCanonicalUnchecked("//other:local").getPackageIdentifier();
    var graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    var packageEntry = graph.getIfPresent(packageKey);

    write("other/BUILD", "exports_files(['local'])", "filegroup(name = 'new')");
    assertDownloadSuccess("//scope:subject");
    assertThat(graph.getIfPresent(packageKey)).isSameInstanceAs(packageEntry);
    assertThat(packageEntry.isDirty()).isTrue();
    var label = parseCanonicalUnchecked("//scope:subject");
    var cachedKey =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getCacheHits().stream()
            .filter(key -> key instanceof ConfiguredTargetKey ct && ct.getLabel().equals(label))
            .findFirst()
            .orElseThrow();
    assertThat(graph.getIfPresent(cachedKey).toValue()).isInstanceOf(DeserializedSkyValue.class);

    // Request both packages: queries discard still-dirty nodes outside their scope.
    if (aquery) {
      runtimeWrapper.runAqueryExprCommand("deps(//scope:both)", "//scope:all");
    } else {
      runtimeWrapper.runCqueryExprCommand("deps(//scope:both)", "//scope:all");
    }

    var localValue = graph.getIfPresent(cachedKey).getValue();
    assertThat(localValue).isNotNull();
    assertThat(localValue).isNotInstanceOf(DeserializedSkyValue.class);
    assertThat(graph.getIfPresent(packageKey)).isSameInstanceAs(packageEntry);
  }

  @Test
  public void buildCommand_uploadsFrontierBytesWithUploadMode() throws Exception {
    setupScenarioWithAspects();
    assertUploadSuccess("//bar:one");

    var listener = getCommandEnvironment().getRemoteAnalysisCachingEventListener();
    assertThat(listener.getSerializedKeysCount()).isAtLeast(9); // for Bazel
    assertThat(listener.getSkyfunctionCounts().count(SkyFunctions.CONFIGURED_TARGET))
        .isAtLeast(9); // for Bazel
  }

  @Test
  public void buildCommand_withWriteFailure_reportsErrorAndCompletes() throws Exception {
    setupScenarioWithAspects();

    failingStore.failNextPut();

    addOptions(UPLOAD_MODE_OPTION);
    var thrown = assertThrows(AbruptExitException.class, () -> buildTarget("//bar:one"));
    assertThat(thrown)
        .hasMessageThat()
        .contains("Simulated write failure for " + failingStore.getFailedKey());

    assertThat(failingStore.getFailCounter()).isEqualTo(1);
    assertContainsEvent("Simulated write failure for " + failingStore.getFailedKey());
  }

  @Test
  public void treeArtifactCoverage() throws Exception {
    write(
        "tree/rule.bzl",
        """
        def _tree_rule_impl(ctx):
            dir = ctx.actions.declare_directory(ctx.attr.name + "_dir")
            ctx.actions.run_shell(
                outputs = [dir],
                command = "mkdir -p " + dir.path + " && touch " + dir.path + "/file.txt",
            )
            return [DefaultInfo(files = depset([dir]))]
        tree_rule = rule(
            implementation = _tree_rule_impl,
        )
        """);
    write(
        "tree/BUILD",
        """
        load("//tree:rule.bzl", "tree_rule")
        tree_rule(name = "my_tree")
        """);

    assertUploadSuccess("//tree:my_tree");

    var serializedKeys =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();

    // Verify that tree artifacts (which are DerivedArtifacts that do not have generating action
    // keys)
    // are serialized as TreeArtifact keys in the frontier.
    var treeArtifacts = filterKeys(serializedKeys, TreeArtifact.class);
    assertThat(treeArtifacts).isNotEmpty();
    assertThat(
            treeArtifacts.stream()
                .map(TreeArtifact::getRootRelativePathString)
                .collect(toImmutableList()))
        .containsExactly("tree/my_tree_dir");
  }

  @Test
  public void constantMetadataCoverage() throws Exception {
    write("bar/BUILD", "genrule(name = 'one', outs = ['one.txt'], cmd = 'touch $@')");
    var unused = getSkyframeExecutor().getWorkspaceStatusArtifacts(events.reporter());
    assertUploadSuccess("//bar:one");

    var serializedKeys =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();

    // Verify that unshareable artifacts/actions (such as constant metadata artifacts or status
    // actions) are NOT serialized, but they DO exist in the graph (proving we actually filtered
    // them).
    assertThat(
            filterKeys(serializedKeys, ActionLookupData.class).stream()
                .filter(data -> !data.valueIsShareable())
                .collect(toImmutableList()))
        .isEmpty();
    assertThat(
            filterKeys(
                    getSkyframeExecutor().getEvaluator().getInMemoryGraph().getValues().keySet(),
                    ActionLookupData.class)
                .stream()
                .filter(data -> !data.valueIsShareable())
                .collect(toImmutableList()))
        .containsExactly(ActionLookupData.create(WorkspaceStatusValue.BUILD_INFO_KEY, 0));
  }
}
