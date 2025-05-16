// Copyright 2025 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/ARIANE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.LongVersionGetterTestInjection.injectVersionGetterForTesting;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.concurrent.RequestBatcher;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SerializationModule;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AnalysisCacheServiceInvalidationIntegrationTest
    extends BuildIntegrationTestCase {

  private final LongVersionGetter versionGetter = mock(LongVersionGetter.class);

  /**
   * A unique instance of the fingerprint value service per test case.
   *
   * <p>This ensures that test cases don't share state. The instance will then last the lifetime of
   * the test case, regardless of the number of command invocations.
   */
  private final FingerprintValueService service = FingerprintValueService.createForTesting();

  private class ModuleWithOverrides extends SerializationModule {
    @Override
    protected RemoteAnalysisCachingServicesSupplier getAnalysisCachingServicesSupplier() {
      return new TestServicesSupplier(service);
    }
  }

  private static final TrivialKey KEY_1 = new TrivialKey("key1");
  private static final TrivialKey KEY_2 = new TrivialKey("key2");

  private static class TestServicesSupplier implements RemoteAnalysisCachingServicesSupplier {
    private final ListenableFuture<FingerprintValueService> wrappedService;
    // Store the fake client data here.
    public final Map<ByteString, ByteString> fakeAnalysisCacheServiceResponses =
        new ConcurrentHashMap<>();

    private TestServicesSupplier(FingerprintValueService fingerprintValueService) {
      this.wrappedService = immediateFuture(fingerprintValueService);
    }

    @Override
    public ListenableFuture<FingerprintValueService> getFingerprintValueService() {
      return wrappedService;
    }

    @Override
    public ListenableFuture<RequestBatcher<ByteString, ByteString>> getAnalysisCacheClient() {
      // Return a future containing the fake client.
      return immediateFuture(createFakeAnalysisCacheClient());
    }

    /** Creates a fake RequestBatcher that uses the provided map as its data source. */
    private RequestBatcher<ByteString, ByteString> createFakeAnalysisCacheClient() {
      return RequestBatcher.create(
          commonPool(),
          keys ->
              immediateFuture(
                  keys.stream()
                      .map(
                          key ->
                              fakeAnalysisCacheServiceResponses.getOrDefault(
                                  key, ByteString.empty()))
                      .collect(toImmutableList())),
          /* maxBatchSize= */ 1,
          /* maxConcurrentRequests= */ 1);
    }
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(new ModuleWithOverrides());
  }

  @Before
  public void setup() throws Exception {
    injectVersionGetterForTesting(versionGetter);
    setupGenruleGraph();
    addOptions("--experimental_remote_analysis_cache_mode=upload");
    buildTarget("//A");

    getSkyframeExecutor().resetEvaluator();

    TestServicesSupplier supplier =
        (TestServicesSupplier)
            getCommandEnvironment().getBlazeWorkspace().remoteAnalysisCachingServicesSupplier();
    supplier.fakeAnalysisCacheServiceResponses.clear();
  }

  @Test
  public void testLookupKeysForInvalidation_noKeysToLookup() throws Exception {
    var provider =
        getCommandEnvironment()
            .getSkyframeExecutor()
            .getRemoteAnalysisCachingDependenciesProvider();
    var keys = provider.lookupKeysToInvalidate(ImmutableSet.of());
    assertThat(keys).isEmpty();
  }

  @Test
  public void testLookupKeysForInvalidation_cacheMiss() throws Exception {
    var provider =
        getCommandEnvironment()
            .getSkyframeExecutor()
            .getRemoteAnalysisCachingDependenciesProvider();
    var keys = provider.lookupKeysToInvalidate(ImmutableSet.of(KEY_1));
    assertThat(keys).containsExactly(KEY_1);
  }

  @Test
  public void testLookupKeysForInvalidation_cacheHit() throws Exception {
    var provider =
        getCommandEnvironment()
            .getSkyframeExecutor()
            .getRemoteAnalysisCachingDependenciesProvider();
    TestServicesSupplier supplier =
        (TestServicesSupplier)
            getCommandEnvironment().getBlazeWorkspace().remoteAnalysisCachingServicesSupplier();
    supplier.fakeAnalysisCacheServiceResponses.put(
        ByteString.copyFrom(getFingerprint(provider, KEY_1).toBytes()),
        ByteString.copyFromUtf8("value"));
    var keys = provider.lookupKeysToInvalidate(ImmutableSet.of(KEY_1));
    assertThat(keys).isEmpty();
  }

  @Test
  public void testLookupKeysForInvalidation_someCacheHitsSomeCacheMisses() throws Exception {
    var provider =
        getCommandEnvironment()
            .getSkyframeExecutor()
            .getRemoteAnalysisCachingDependenciesProvider();
    TestServicesSupplier supplier =
        (TestServicesSupplier)
            getCommandEnvironment().getBlazeWorkspace().remoteAnalysisCachingServicesSupplier();
    supplier.fakeAnalysisCacheServiceResponses.put(
        ByteString.copyFrom(getFingerprint(provider, KEY_1).toBytes()),
        ByteString.copyFromUtf8("value"));
    var keys = provider.lookupKeysToInvalidate(ImmutableSet.of(KEY_1, KEY_2));
    assertThat(keys).containsExactly(KEY_2);
  }

  // Helper to get the fingerprint for a key using the current provider context.
  private PackedFingerprint getFingerprint(
      RemoteAnalysisCachingDependenciesProvider provider, SkyKey key)
      throws InterruptedException, ExecutionException, SerializationException {
    ObjectCodecs codecs = provider.getObjectCodecs();
    FingerprintValueService fingerprintService = provider.getFingerprintValueService();
    SerializationResult<ByteString> keyBytesResult =
        codecs.serializeMemoizedAndBlocking(fingerprintService, key, /* profileCollector= */ null);
    return fingerprintService.fingerprint(
        provider.getSkyValueVersion().concat(keyBytesResult.getObject().toByteArray()));
  }

  @AutoCodec
  @VisibleForSerialization
  record TrivialKey(String text) implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      throw new UnsupportedOperationException();
    }
  }

  private final void setupGenruleGraph() throws IOException {
    write("A/in.txt", "A");
    write(
        "A/BUILD",
        """
        genrule(
            name = "A",
            srcs = ["in.txt", "//C:C.txt"],
            outs = ["A"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("B/in.txt", "B");
    write(
        "B/BUILD",
        """
        genrule(
            name = "B",
            srcs = ["in.txt", "//C:C.txt"],
            outs = ["B"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("C/in.txt", "C");
    write(
        "C/BUILD",
        """
        genrule(
            name = "C",
            srcs = ["in.txt"],
            outs = ["C.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write(
        "A/PROJECT.scl",
"""
project = {
  "active_directories": {"default": ["A"]},
}
""");
  }
}
