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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping.roundTripWithSkyframe;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.Types;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link StarlarkProviderCodec}. */
@RunWith(JUnit4.class)
public final class StarlarkProviderCodecTest {
  // For legacy reasons, this test exhibits a number of different kinds of providers. Since
  // serialization is only based on serializing the key, and has nothing to do with the provider
  // contents, it might be possible to simplify these tests. They are left mostly intact in case the
  // implementation changes in the future.

  private static final CompressionService COMPRESSION_SERVICE = new CompressionServiceImpl();

  private final BzlLoadValue.Key bzlLoadKey =
      keyForBuild(Label.parseCanonicalUnchecked("//foo.bzl"));
  private final StarlarkProvider.Key providerKey = new StarlarkProvider.Key(bzlLoadKey, "prov");

  private static final StarlarkCallable DUMMY_INIT =
      new StarlarkCallable() {
        @Override
        public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
            throws EvalException {
          return Dict.empty();
        }

        @Override
        public String getName() {
          return "init";
        }

        @Override
        public Location getLocation() {
          return Location.BUILTIN;
        }

        @Override
        public StarlarkType getStarlarkType(StarlarkSemantics semantics) {
          return Types.ANY_CALLABLE;
        }
      };

  @Test
  public void schemaWithDocumentation() throws Exception {
    String documentation = "documentation";
    ImmutableMap<String, String> schemaWithDocumentation =
        ImmutableMap.of(
            "field1", "documentation1", "field2", "documentation2", "field3", "documentation3");
    var provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setDocumentation(documentation)
            .setSchema(schemaWithDocumentation)
            .setInit(DUMMY_INIT)
            .buildExported(providerKey);

    var deserialized =
        (StarlarkProvider)
            roundTripWithSkyframe(COMPRESSION_SERVICE, createFakeResultMap(provider), provider);
    assertThat(deserialized).isSameInstanceAs(provider);
  }

  @Test
  public void schemaWithoutDocumentation() throws Exception {
    String documentation = "documentation";
    ImmutableList<String> fields = ImmutableList.of("a", "b", "c");
    var provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setDocumentation(documentation)
            .setSchema(fields)
            .setInit(DUMMY_INIT)
            .buildExported(providerKey);

    var deserialized =
        (StarlarkProvider)
            roundTripWithSkyframe(COMPRESSION_SERVICE, createFakeResultMap(provider), provider);
    assertThat(deserialized).isSameInstanceAs(provider);
  }

  @Test
  public void nullSchema() throws Exception {
    String documentation = "documentation";
    var provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setDocumentation(documentation)
            .setInit(DUMMY_INIT)
            .buildExported(providerKey);
    var deserialized =
        (StarlarkProvider)
            roundTripWithSkyframe(COMPRESSION_SERVICE, createFakeResultMap(provider), provider);
    assertThat(deserialized).isSameInstanceAs(provider);
  }

  @Test
  public void emptySchema() throws Exception {
    String documentation = "documentation";
    var provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setDocumentation(documentation)
            .setSchema(ImmutableList.of())
            .setInit(DUMMY_INIT)
            .buildExported(providerKey);
    var deserialized =
        (StarlarkProvider)
            roundTripWithSkyframe(COMPRESSION_SERVICE, createFakeResultMap(provider), provider);
    assertThat(deserialized).isSameInstanceAs(provider);
  }

  @Test
  public void noInit() throws Exception {
    String documentation = "documentation";
    ImmutableList<String> fields = ImmutableList.of("a", "b", "c");
    var provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setDocumentation(documentation)
            .setSchema(fields)
            .buildExported(providerKey);
    var deserialized =
        (StarlarkProvider)
            roundTripWithSkyframe(COMPRESSION_SERVICE, createFakeResultMap(provider), provider);
    assertThat(deserialized).isSameInstanceAs(provider);
  }

  @Test
  public void noDocumentation() throws Exception {
    ImmutableList<String> fields = ImmutableList.of("a", "b", "c");
    var provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(fields)
            .setInit(DUMMY_INIT)
            .buildExported(providerKey);
    var deserialized =
        (StarlarkProvider)
            roundTripWithSkyframe(COMPRESSION_SERVICE, createFakeResultMap(provider), provider);
    assertThat(deserialized).isSameInstanceAs(provider);
  }

  @Test
  public void unexportedProvider_cannotBeSerialized() throws Exception {
    var unexportedProvider =
        StarlarkProvider.builder(Location.BUILTIN)
            .buildWithIdentityToken(SymbolGenerator.createTransient().generate());
    ObjectCodecs objectCodecs = new ObjectCodecs();
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> objectCodecs.serializeMemoized(unexportedProvider));
    assertThat(e).hasMessageThat().contains("Cannot serialize unexported Starlark provider");
  }

  private EnvironmentForUtilities.ResultProvider createFakeResultMap(StarlarkProvider provider) {
    var module = Module.create();
    module.setGlobal(providerKey.getExportedName(), provider);
    return ImmutableMap.of(
            bzlLoadKey,
            new BzlLoadValue(
                module,
                /* transitiveDigest= */ new byte[0],
                BzlVisibility.PUBLIC,
                /* recordedRepoMappings= */ ImmutableTable.of()))
        ::get;
  }
}
