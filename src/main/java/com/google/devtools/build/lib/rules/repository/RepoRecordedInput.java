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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Represents a "recorded input" of a repo fetch. We define the "input" of a repo fetch as any
 * entity that could affect the output of the repo fetch (i.e. the repo contents). A "recorded
 * input" is thus any input we can record during the fetch and thus know about only after the fetch.
 * This contrasts with "predeclared inputs", which are known before fetching the repo, and
 * "undiscoverable inputs", which are used during the fetch but is not recorded or recordable.
 *
 * <p>Recorded inputs are of particular interest, since in order to determine whether a fetched repo
 * is still up-to-date, the identity of all recorded inputs need to be stored in addition to their
 * values. This contrasts with predeclared inputs; the whole set of predeclared inputs are known
 * before the fetch, so we can simply hash all predeclared input values.
 *
 * <p>Recorded inputs and their values are stored in <i>marker files</i> for repos. Each recorded
 * input is stored as a string, with a prefix denoting its type, followed by a colon, and then the
 * information identifying that specific input.
 */
public abstract class RepoRecordedInput implements Comparable<RepoRecordedInput> {
  /** Represents a parser for a specific type of recorded inputs. */
  public abstract static class Parser {
    /**
     * The prefix that identifies the type of the recorded inputs: for example, the {@code ENV} part
     * of {@code ENV:MY_ENV_VAR}.
     */
    public abstract String getPrefix();

    /**
     * Parses a recorded input from the post-colon substring that identifies the specific input: for
     * example, the {@code MY_ENV_VAR} part of {@code ENV:MY_ENV_VAR}.
     */
    public abstract RepoRecordedInput parse(String s, Path baseDirectory);
  }

  private static final Comparator<RepoRecordedInput> COMPARATOR =
      Comparator.comparing((RepoRecordedInput rri) -> rri.getParser().getPrefix())
          .thenComparing(RepoRecordedInput::toStringInternal);

  /**
   * Parses a recorded input from its string representation.
   *
   * @param s the string representation
   * @param baseDirectory the path to a base directory that any filesystem paths should be resolved
   *     relative to
   * @return The parsed recorded input object, or {@code null} if the string representation is
   *     invalid
   */
  @Nullable
  public static RepoRecordedInput parse(String s, Path baseDirectory) {
    List<String> parts = Splitter.on(':').limit(2).splitToList(s);
    for (Parser parser : new Parser[] {File.PARSER, EnvVar.PARSER, RecordedRepoMapping.PARSER}) {
      if (parts.get(0).equals(parser.getPrefix())) {
        return parser.parse(parts.get(1), baseDirectory);
      }
    }
    return null;
  }

  @Override
  public final String toString() {
    return getParser().getPrefix() + ":" + toStringInternal();
  }

  @Override
  public int compareTo(RepoRecordedInput o) {
    return COMPARATOR.compare(this, o);
  }

  /**
   * Returns the post-colon substring that identifies the specific input: for example, the {@code
   * MY_ENV_VAR} part of {@code ENV:MY_ENV_VAR}.
   */
  abstract String toStringInternal();

  /** Returns the parser object for this type of recorded inputs. */
  public abstract Parser getParser();

  /** Returns the {@link SkyKey} that is necessary to determine {@link #isUpToDate}. */
  public abstract SkyKey getSkyKey();

  /**
   * Returns whether the given {@code oldValue} is still up-to-date for this recorded input. This
   * method can assume that {@link #getSkyKey()} is already evaluated; it can request further
   * Skyframe evaluations, and if any values are missing, this method can return any value (doesn't
   * matter what) and will be reinvoked after a Skyframe restart.
   */
  public abstract boolean isUpToDate(Environment env, @Nullable String oldValue)
      throws InterruptedException;

  /** Represents a file input accessed during the repo fetch. */
  public abstract static class File extends RepoRecordedInput {
    static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "FILE";
          }

          @Override
          public RepoRecordedInput parse(String s, Path baseDirectory) {
            if (LabelValidator.isAbsolute(s)) {
              return new LabelFile(Label.parseCanonicalUnchecked(s));
            }
            Path path = baseDirectory.getRelative(s);
            return new AbsolutePathFile(
                RootedPath.toRootedPath(
                    Root.fromPath(path.getParentDirectory()),
                    PathFragment.create(path.getBaseName())));
          }
        };

    @Override
    public Parser getParser() {
      return PARSER;
    }
  }

  /** Represents a file input accessed during the repo fetch that is addressable by a label. */
  public static final class LabelFile extends File {
    final Label label;

    public LabelFile(Label label) {
      this.label = label;
    }

    @Override
    String toStringInternal() {
      return label.toString();
    }

    @Override
    public SkyKey getSkyKey() {
      return PackageLookupValue.key(label.getPackageIdentifier());
    }

    @Override
    public boolean isUpToDate(Environment env, @Nullable String oldValue)
        throws InterruptedException {
      PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(getSkyKey());
      if (pkgLookupValue == null || !pkgLookupValue.packageExists()) {
        return false;
      }
      RootedPath rootedPath =
          RootedPath.toRootedPath(pkgLookupValue.getRoot(), label.toPathFragment());
      SkyKey fileKey = FileValue.key(rootedPath);
      try {
        FileValue fileValue = (FileValue) env.getValueOrThrow(fileKey, IOException.class);
        if (fileValue == null || !fileValue.isFile() || fileValue.isSpecialFile()) {
          return false;
        }
        return oldValue.equals(RepositoryFunction.fileValueToMarkerValue(fileValue));
      } catch (IOException e) {
        return false;
      }
    }
  }

  /**
   * Represents a file input accessed during the repo fetch that is <i>not</i> addressable by a
   * label. This most likely means that it's outside any known Bazel workspace.
   */
  public static final class AbsolutePathFile extends File {
    final RootedPath path;

    public AbsolutePathFile(RootedPath path) {
      this.path = path;
    }

    @Override
    String toStringInternal() {
      return path.asPath().getPathString();
    }

    @Override
    public SkyKey getSkyKey() {
      return FileValue.key(path);
    }

    @Override
    public boolean isUpToDate(Environment env, @Nullable String oldValue)
        throws InterruptedException {
      try {
        FileValue fileValue = (FileValue) env.getValueOrThrow(getSkyKey(), IOException.class);
        if (fileValue == null || !fileValue.isFile() || fileValue.isSpecialFile()) {
          return false;
        }
        return oldValue.equals(RepositoryFunction.fileValueToMarkerValue(fileValue));
      } catch (IOException e) {
        return false;
      }
    }
  }

  /** Represents an environment variable accessed during the repo fetch. */
  public static final class EnvVar extends RepoRecordedInput {
    static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "ENV";
          }

          @Override
          public RepoRecordedInput parse(String s, Path baseDirectory) {
            return new EnvVar(s);
          }
        };

    final String name;

    public EnvVar(String name) {
      this.name = name;
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    String toStringInternal() {
      return name;
    }

    @Override
    public SkyKey getSkyKey() {
      return ActionEnvironmentFunction.key(name);
    }

    @Override
    public boolean isUpToDate(Environment env, @Nullable String oldValue)
        throws InterruptedException {
      String v = PrecomputedValue.REPO_ENV.get(env).get(name);
      if (v == null) {
        v = ((ClientEnvironmentValue) env.getValue(getSkyKey())).getValue();
      }
      // Note that `oldValue` can be null if the env var was not set.
      return Objects.equals(oldValue, v);
    }
  }

  /** Represents a repo mapping entry that was used during the repo fetch. */
  public static final class RecordedRepoMapping extends RepoRecordedInput {
    static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "REPO_MAPPING";
          }

          @Override
          public RepoRecordedInput parse(String s, Path baseDirectory) {
            List<String> parts = Splitter.on(',').limit(2).splitToList(s);
            return new RecordedRepoMapping(
                RepositoryName.createUnvalidated(parts.get(0)), parts.get(1));
          }
        };

    final RepositoryName sourceRepo;
    final String apparentName;

    public RecordedRepoMapping(RepositoryName sourceRepo, String apparentName) {
      this.sourceRepo = sourceRepo;
      this.apparentName = apparentName;
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    String toStringInternal() {
      return sourceRepo.getName() + ',' + apparentName;
    }

    @Override
    public SkyKey getSkyKey() {
      return RepositoryMappingValue.key(sourceRepo);
    }

    @Override
    public boolean isUpToDate(Environment env, @Nullable String oldValue)
        throws InterruptedException {
      RepositoryMappingValue repoMappingValue = (RepositoryMappingValue) env.getValue(getSkyKey());
      return repoMappingValue != RepositoryMappingValue.NOT_FOUND_VALUE
          && RepositoryName.createUnvalidated(oldValue)
              .equals(repoMappingValue.getRepositoryMapping().get(apparentName));
    }
  }
}
