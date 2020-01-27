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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.ParsedAndroidData.CombiningConsumer;
import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;
import com.google.devtools.build.android.ParsedAndroidData.OverwritableConsumer;
import com.google.devtools.build.android.resources.Visibility;
import com.google.devtools.build.android.xml.Namespaces;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Build for ParsedAndroidData instance.
 */
public class ParsedAndroidDataBuilder {

  private final Path defaultRoot;
  private final FullyQualifiedName.Factory fqnFactory;
  private final Map<DataKey, DataResource> overwrite = new HashMap<>();
  private final Map<DataKey, DataResource> combine = new HashMap<>();
  private final Map<DataKey, DataAsset> assets = new HashMap<>();
  private final Set<MergeConflict> conflicts = new HashSet<>();

  public ParsedAndroidDataBuilder(
      @Nullable Path root, @Nullable FullyQualifiedName.Factory fqnFactory) {
    this.defaultRoot = root;
    this.fqnFactory = fqnFactory;
  }

  public static ParsedAndroidData empty() {
    return ParsedAndroidData.of(
        ImmutableSet.<MergeConflict>of(),
        ImmutableMap.<DataKey, DataResource>of(),
        ImmutableMap.<DataKey, DataResource>of(),
        ImmutableMap.<DataKey, DataAsset>of());
  }

  public static ParsedAndroidDataBuilder buildOn(
      Path defaultRoot, FullyQualifiedName.Factory fqnFactory) {
    return new ParsedAndroidDataBuilder(defaultRoot, fqnFactory);
  }

  public static ParsedAndroidDataBuilder buildOn(FullyQualifiedName.Factory fqnFactory) {
    return buildOn(null, fqnFactory);
  }

  public static ParsedAndroidDataBuilder buildOn(Path defaultRoot) {
    return buildOn(defaultRoot, null);
  }

  public static ParsedAndroidDataBuilder builder() {
    return buildOn(null, null);
  }

  public ParsedAndroidDataBuilder overwritable(DataEntry... resourceBuilders) {
    OverwritableConsumer<DataKey, DataResource> consumer =
        new OverwritableConsumer<>(overwrite, conflicts);
    for (DataEntry resourceBuilder : resourceBuilders) {
      resourceBuilder.accept(fqnFactory, defaultRoot, consumer);
    }
    return this;
  }

  public ParsedAndroidDataBuilder combining(DataEntry... resourceBuilders) {
    CombiningConsumer consumer = new CombiningConsumer(combine);
    for (DataEntry resourceBuilder : resourceBuilders) {
      resourceBuilder.accept(fqnFactory, defaultRoot, consumer);
    }
    return this;
  }

  public ParsedAndroidDataBuilder assets(DataEntry... assetBuilders) {
    OverwritableConsumer<DataKey, DataAsset> consumer =
        new OverwritableConsumer<>(assets, conflicts);
    for (DataEntry assetBuilder : assetBuilders) {
      assetBuilder.accept(defaultRoot, consumer);
    }
    return this;
  }

  public static FileResourceBuilder file(String rawKey) {
    return new FileResourceBuilder(rawKey);
  }

  public static FileResourceBuilder file() {
    return new FileResourceBuilder(null);
  }

  public static XmlResourceBuilder xml(String rawKey) {
    return new XmlResourceBuilder(rawKey);
  }

  public ParsedAndroidData build() {
    return ParsedAndroidData.of(
        ImmutableSet.copyOf(conflicts),
        ImmutableMap.copyOf(overwrite),
        ImmutableMap.copyOf(combine),
        ImmutableMap.copyOf(assets));
  }

  static class FileResourceBuilder {
    private String rawKey;
    private Path root;

    FileResourceBuilder(@Nullable String rawKey) {
      this.rawKey = rawKey;
    }

    FileResourceBuilder root(Path root) {
      this.root = root;
      return this;
    }

    Path chooseRoot(Path defaultRoot) {
      if (defaultRoot != null) {
        return defaultRoot;
      }
      if (root != null) {
        return root;
      }
      throw new IllegalStateException(
          "the default root and asset root are null! A root is required!");
    }

    DataEntry source(final DataSource source) {
      return new DataEntry() {
        @Override
        void accept(
            @Nullable Factory factory,
            @Nullable Path root,
            KeyValueConsumer<DataKey, DataResource> consumer) {
          consumer.accept(
              factory.parse(rawKey),
              DataValueFile.of(
                  Visibility.UNKNOWN, source, /*fingerprint=*/ null, /*rootXmlNode=*/ null));
        }

        @Override
        void accept(@Nullable Path defaultRoot, KeyValueConsumer<DataKey, DataAsset> target) {
          target.accept(
              RelativeAssetPath.Factory.of(chooseRoot(defaultRoot).resolve("assets"))
                  .create(source.getPath()),
              DataValueFile.of(
                  Visibility.UNKNOWN, source, /*fingerprint=*/ null, /*rootXmlNode=*/ null));
        }
      };
    }

    DataEntry source(final String path) {
      return new DataEntry() {
        @Override
        public void accept(
            FullyQualifiedName.Factory factory,
            Path defaultRoot,
            KeyValueConsumer<DataKey, DataResource> consumer) {
          Path res = chooseRoot(defaultRoot).resolve("res");
          consumer.accept(factory.parse(rawKey), DataValueFile.of(res.resolve(path)));
        }

        @Override
        public void accept(
            @Nullable Path defaultRoot, KeyValueConsumer<DataKey, DataAsset> consumer) {
          Path assets = chooseRoot(defaultRoot).resolve("assets");
          Path fullPath = assets.resolve(path);
          consumer.accept(
              RelativeAssetPath.Factory.of(assets).create(fullPath), DataValueFile.of(fullPath));
        }
      };
    }
  }

  static class XmlResourceBuilder {
    private final String rawFqn;
    private Path root;
    private final Map<String, String> prefixToUri = new LinkedHashMap<>();

    XmlResourceBuilder(String rawFqn) {
      this(rawFqn, null);
    }

    XmlResourceBuilder(String rawFqn, Path root) {
      this.rawFqn = rawFqn;
      this.root = root;
    }

    XmlResourceBuilder source(final String path) {
      return new XmlResourceBuilder(rawFqn, root) {
        @Override
        public DataEntry value(final XmlResourceValue value) {
          return new DataEntry() {
            @Override
            public void accept(
                FullyQualifiedName.Factory factory,
                Path defaultRoot,
                KeyValueConsumer<DataKey, DataResource> consumer) {
              Path res = (root == null ? defaultRoot : root).resolve("res");
              consumer.accept(
                  factory.parse(rawFqn),
                  DataResourceXml.createWithNamespaces(
                      res.resolve(path), value, Namespaces.from(prefixToUri)));
            }
          };
        }
      };
    }

    XmlResourceBuilder source(final DataSource dataSource) {
      return new XmlResourceBuilder(rawFqn, root) {
        @Override
        public DataEntry value(final XmlResourceValue value) {
          return new DataEntry() {
            @Override
            public void accept(
                FullyQualifiedName.Factory factory,
                Path defaultRoot,
                KeyValueConsumer<DataKey, DataResource> consumer) {
              consumer.accept(
                  factory.parse(rawFqn),
                  DataResourceXml.createWithNamespaces(
                      dataSource, value, Namespaces.from(prefixToUri)));
            }
          };
        }
      };
    }

    XmlResourceBuilder root(Path root) {
      this.root = root;
      return this;
    }

    XmlResourceBuilder namespace(String prefix, String uri) {
      prefixToUri.put(prefix, uri);
      return this;
    }

    DataEntry value(final XmlResourceValue value) {
      throw new UnsupportedOperationException("A source must be declared!");
    }
  }

  abstract static class DataEntry {
    void accept(
        @Nullable FullyQualifiedName.Factory factory,
        @Nullable Path root,
        KeyValueConsumer<DataKey, DataResource> consumer) {
      throw new UnsupportedOperationException("assets cannot be resources!");
    }

    void accept(@Nullable Path root, KeyValueConsumer<DataKey, DataAsset> target) {
      throw new UnsupportedOperationException("xml resources cannot be assets!");
    }
  }
}
