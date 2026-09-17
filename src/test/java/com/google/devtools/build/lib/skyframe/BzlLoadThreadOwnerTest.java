// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyframeDependencyException;
import com.google.devtools.build.lib.skyframe.serialization.SkyframeLookupContinuation;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import com.google.protobuf.ByteString;
import java.util.concurrent.ExecutionException;
import net.starlark.java.eval.Module;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for BzlLoadFunction. */
@RunWith(JUnit4.class)
public class BzlLoadThreadOwnerTest extends BuildViewTestCase {

  private final ObjectCodecs objectCodecs = new ObjectCodecs();
  private final FingerprintValueService fingerprintValueService =
      FingerprintValueService.createForTesting();

  @Test
  public void owner_detectsDirectChange() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/lib.bzl",
        """
        def f():
            return 1
        """);
    BzlLoadThreadOwner oldOwner = getOwner("//test:lib.bzl");
    ByteString oldOwnerSerialized = objectCodecs.serializeMemoized(oldOwner);

    scratch.overwriteFile(
        "test/lib.bzl",
        """
        def f():
            return 2
        """);
    invalidatePackages();
    BzlLoadThreadOwner newOwner = getOwner("//test:lib.bzl");

    assertThat(oldOwner.key()).isEqualTo(newOwner.key());
    assertThat(oldOwner).isNotEqualTo(newOwner);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> deserializeWithSkyframe(oldOwnerSerialized));
    assertThat(exception)
        .hasCauseThat()
        .hasMessageThat()
        .contains("Cannot retrieve a zombie module");
  }

  @Test
  public void owner_detectsTransitiveChange() throws Exception {
    scratch.file("test/BUILD");
    scratch.file("test/dep.bzl", "x = 1");
    scratch.file(
        "test/lib.bzl",
        """
        load("dep.bzl", "x")
        def f():
            return x
        """);
    BzlLoadThreadOwner oldOwner = getOwner("//test:lib.bzl");
    ByteString oldOwnerSerialized = objectCodecs.serializeMemoized(oldOwner);

    scratch.overwriteFile("test/dep.bzl", "x = 2");
    invalidatePackages();
    BzlLoadThreadOwner newOwner = getOwner("//test:lib.bzl");

    assertThat(oldOwner.key()).isEqualTo(newOwner.key());
    assertThat(oldOwner).isNotEqualTo(newOwner);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> deserializeWithSkyframe(oldOwnerSerialized));
    assertThat(exception)
        .hasCauseThat()
        .hasMessageThat()
        .contains("Cannot retrieve a zombie module");
  }

  @Test
  public void owner_ignoresUnrelatedChange() throws Exception {
    scratch.file("test/BUILD", "load('//test:unrelated.bzl', 'x')");
    scratch.file("test/unrelated.bzl", "x = 1");
    scratch.file(
        "test/lib.bzl",
        """
        def f():
            return 42
        """);
    getTarget("//test:BUILD");
    BzlLoadThreadOwner oldOwner = getOwner("//test:lib.bzl");
    ByteString oldOwnerSerialized = objectCodecs.serializeMemoized(oldOwner);

    scratch.overwriteFile("test/unrelated.bzl", "x = 2");
    invalidatePackages();
    getTarget("//test:BUILD");
    BzlLoadThreadOwner newOwner = getOwner("//test:lib.bzl");

    assertThat(oldOwner).isEqualTo(newOwner);
    BzlLoadThreadOwner oldOwnerDeserialized = deserializeWithSkyframe(oldOwnerSerialized);
    assertThat(oldOwnerDeserialized).isEqualTo(oldOwner);
  }

  private BzlLoadValue.Key getKey(String label) {
    return BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked(label));
  }

  private BzlLoadValue getBzlLoadValue(BzlLoadValue.Key key) throws InterruptedException {
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), key, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      fail(result.getError(key).getException().getMessage());
    }
    return result.get(key);
  }

  private BzlLoadThreadOwner getOwner(String label) throws InterruptedException {
    BzlLoadValue.Key key = getKey(label);
    Module module = getBzlLoadValue(key).getModule();
    // Ensure that the file has been processed by checking its Module for the label field.
    assertThat(Label.parseCanonicalUnchecked(label))
        .isEqualTo(BazelModuleContext.of(module).label());
    return BzlLoadThreadOwner.of(key, module);
  }

  private BzlLoadThreadOwner deserializeWithSkyframe(ByteString serialized)
      throws ExecutionException,
          InterruptedException,
          SerializationException,
          SkyframeDependencyException {
    // Deserialization always returns a future because there is a Skyframe lookup. The future is
    // always done because there are no shared values to wait on.
    SkyframeLookupContinuation continuation =
        (SkyframeLookupContinuation)
            Futures.getDone(
                (ListenableFuture<?>)
                    objectCodecs.deserializeWithSkyframe(fingerprintValueService, serialized));
    ListenableFuture<?> resultFuture =
        continuation.process(
            new EnvironmentForUtilities(
                // The only Skyframe lookup our deserializer needs is for one BzlLoadValue.
                key -> {
                  try {
                    return getBzlLoadValue((BzlLoadValue.Key) key);
                  } catch (InterruptedException e) {
                    throw new AssertionError(e);
                  }
                }));
    return (BzlLoadThreadOwner) Futures.getDone(resultFuture);
  }
}
