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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Iterator;
import java.util.Map;

/** Builder for creating manifest merger actions. */
public class ManifestMergerActionBuilder {

  private Artifact manifest;
  private Map<Artifact, Label> mergeeManifests;
  private boolean isLibrary;
  private Map<String, String> manifestValues;
  private String customPackage;
  private Artifact manifestOutput;
  private Artifact logOut;

  public ManifestMergerActionBuilder setManifest(Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  public ManifestMergerActionBuilder setMergeeManifests(Map<Artifact, Label> mergeeManifests) {
    this.mergeeManifests = ImmutableMap.copyOf(mergeeManifests);
    return this;
  }

  public ManifestMergerActionBuilder setLibrary(boolean isLibrary) {
    this.isLibrary = isLibrary;
    return this;
  }

  public ManifestMergerActionBuilder setManifestValues(Map<String, String> manifestValues) {
    this.manifestValues = manifestValues;
    return this;
  }

  public ManifestMergerActionBuilder setCustomPackage(String customPackage) {
    this.customPackage = customPackage;
    return this;
  }

  public ManifestMergerActionBuilder setManifestOutput(Artifact manifestOutput) {
    this.manifestOutput = manifestOutput;
    return this;
  }

  public ManifestMergerActionBuilder setLogOut(Artifact logOut) {
    this.logOut = logOut;
    return this;
  }

  public void build(AndroidDataContext dataContext) {
    BusyBoxActionBuilder builder =
        BusyBoxActionBuilder.create(dataContext, "MERGE_MANIFEST")
            .maybeAddInput("--manifest", manifest);

    if (mergeeManifests != null) {
      builder.maybeAddInput(
          "--mergeeManifests",
          mapToDictionaryString(
              mergeeManifests, Artifact::getExecPathString, /* valueConverter = */ null),
          mergeeManifests.keySet());
    }

    builder
        .maybeAddFlag("--mergeType", isLibrary)
        .maybeAddFlag("LIBRARY", isLibrary)
        .maybeAddFlag("--manifestValues", mapToDictionaryString(manifestValues))
        .maybeAddFlag("--customPackage", customPackage)
        .addOutput("--manifestOutput", manifestOutput)
        .maybeAddOutput("--log", logOut)
        .buildAndRegister("Merging manifest", "ManifestMerger");
  }

  private static final Function<String, String> ESCAPER =
      (String value) -> value.replace(":", "\\:").replace(",", "\\,");

  private <K, V> String mapToDictionaryString(Map<K, V> map) {
    return mapToDictionaryString(map, Functions.toStringFunction(), Functions.toStringFunction());
  }

  private <K, V> String mapToDictionaryString(
      Map<K, V> map,
      Function<? super K, String> keyConverter,
      Function<? super V, String> valueConverter) {
    if (map == null) {
      return null;
    }
    if (keyConverter == null) {
      keyConverter = Functions.toStringFunction();
    }
    if (valueConverter == null) {
      valueConverter = Functions.toStringFunction();
    }

    StringBuilder sb = new StringBuilder();
    Iterator<Map.Entry<K, V>> iter = map.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<K, V> entry = iter.next();
      sb.append(Functions.compose(ESCAPER, keyConverter).apply(entry.getKey()));
      sb.append(':');
      sb.append(Functions.compose(ESCAPER, valueConverter).apply(entry.getValue()));
      if (iter.hasNext()) {
        sb.append(',');
      }
    }
    return sb.toString();
  }
}
