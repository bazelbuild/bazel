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
package com.google.devtools.build.android;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility for building {@link UnvalidatedAndroidData}, {@link ParsedAndroidData},
 * {@link DependencyAndroidData} and {@link MergedAndroidData}.
 */
public class AndroidDataBuilder {
  /** Templates for resource files generation. */
  public enum ResourceType {
    VALUE {
      @Override
      public String create(String... lines) {
        return String.format(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<resources>%s</resources>",
            Joiner.on("\n").join(lines));
      }
    },
    LAYOUT {
      @Override
      public String create(String... lines) {
        return String.format(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
                + "<LinearLayout xmlns:android=\"http://schemas.android.com/apk/res/android\""
                + " android:layout_width=\"fill_parent\""
                + " android:layout_height=\"fill_parent\">%s</LinearLayout>",
            Joiner.on("\n").join(lines));
      }
    },
    UNFORMATTED {
      @Override
      public String create(String... lines) {
        return String.format(Joiner.on("\n").join(lines));
      }
    };

    public abstract String create(String... lines);
  }

  public static AndroidDataBuilder of(Path root) {
    return new AndroidDataBuilder(root);
  }

  private final Path root;
  private final Path assetDir;
  private final Path resourceDir;
  private Map<Path, String> filesToWrite = new HashMap<>();
  private Map<Path, Path> filesToCopy = new HashMap<>();
  private Path manifest;
  private Path rTxt;

  private AndroidDataBuilder(Path root) {
    this.root = root;
    assetDir = root.resolve("assets");
    resourceDir = root.resolve("res");
  }

  @CanIgnoreReturnValue
  public AndroidDataBuilder addResource(
      String path, AndroidDataBuilder.ResourceType template, String... lines) {
    filesToWrite.put(resourceDir.resolve(path), template.create(lines));
    return this;
  }

  @CanIgnoreReturnValue
  public AndroidDataBuilder addValuesWithAttributes(
      String path, Map<String, String> attributes, String... lines) {
    ImmutableList.Builder<String> attributeBuilder = ImmutableList.builder();
    for (Map.Entry<String, String> attribute : attributes.entrySet()) {
      if (attribute.getKey() != null && attribute.getValue() != null) {
        attributeBuilder.add(String.format("%s=\"%s\"", attribute.getKey(), attribute.getValue()));
      }
    }
    String fileContents = ResourceType.VALUE.create(lines);
    fileContents = fileContents.replace("<resources>",
        String.format("<resources %s>", Joiner.on(" ").join(attributeBuilder.build())));
    filesToWrite.put(resourceDir.resolve(path), fileContents);
    return this;
  }

  @CanIgnoreReturnValue
  public AndroidDataBuilder addResourceBinary(String path, Path source) {
    final Path target = resourceDir.resolve(path);
    filesToCopy.put(target, source);
    return this;
  }

  @CanIgnoreReturnValue
  public AndroidDataBuilder addAsset(String path, String... lines) {
    filesToWrite.put(assetDir.resolve(path), Joiner.on("\n").join(lines));
    return this;
  }

  public AndroidDataBuilder createManifest(String path, String manifestPackage, String... lines) {
    return createManifest(path, manifestPackage, ImmutableList.<String>of(), lines);
  }

  @CanIgnoreReturnValue
  public AndroidDataBuilder createManifest(
      String path, String manifestPackage, List<String> namespaces, String... lines) {
    this.manifest = root.resolve(path);
    filesToWrite.put(
        manifest,
        String.format(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
                + "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" %s"
                + " package=\"%s\">"
                + "%s</manifest>",
            Joiner.on(" ").join(namespaces),
            manifestPackage,
            Joiner.on("\n").join(lines)));
    return this;
  }

  @CanIgnoreReturnValue
  public AndroidDataBuilder createRTxt(String path, String... lines) {
    this.rTxt = root.resolve(path);
    filesToWrite.put(rTxt, Joiner.on("\n").join(lines));
    return this;
  }

  public UnvalidatedAndroidData buildUnvalidated() throws IOException {
    writeFiles();
    return new UnvalidatedAndroidData(
        ImmutableList.of(resourceDir), ImmutableList.of(assetDir), manifest);
  }

  public ParsedAndroidData buildParsed() throws IOException, MergingException {
    return ParsedAndroidData.from(buildUnvalidated());
  }

  public DependencyAndroidData buildDependency() throws IOException {
    writeFiles();
    return new DependencyAndroidData(
        ImmutableList.of(resourceDir), ImmutableList.of(assetDir), manifest, rTxt, null, null);
  }

  public MergedAndroidData buildMerged() throws IOException {
    writeFiles();
    return new MergedAndroidData(resourceDir, assetDir, manifest);
  }

  private void writeFiles() throws IOException {
    Files.createDirectories(assetDir);
    Files.createDirectories(resourceDir);
    Preconditions.checkNotNull(manifest, "A manifest is required.");
    for (Map.Entry<Path, String> entry : filesToWrite.entrySet()) {
      Files.createDirectories(entry.getKey().getParent());
      Files.write(entry.getKey(), entry.getValue().getBytes(StandardCharsets.UTF_8));
      Preconditions.checkArgument(Files.exists(entry.getKey()));
    }
    for (Map.Entry<Path, Path> entry : filesToCopy.entrySet()) {
      Path target = entry.getKey();
      Path source = entry.getValue();
      Files.createDirectories(target.getParent());
      Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
    }
  }

}
