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
package com.google.devtools.build.android;

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Builder for AAPT command lines, with support for making flags conditional on build tools version
 * and variant type.
 */
public class AaptCommandBuilder {

  private final ImmutableList.Builder<String> flags = new ImmutableList.Builder<>();
  private Revision buildToolsVersion;
  private VariantType variantType;

  public AaptCommandBuilder(Path aapt) {
    flags.add(aapt.toString());
  }

  /** Sets the build tools version to be used for {@link #whenVersionIsAtLeast}. */
  public AaptCommandBuilder forBuildToolsVersion(@Nullable Revision buildToolsVersion) {
    Preconditions.checkState(
        this.buildToolsVersion == null, "A build tools version was already specified.");
    this.buildToolsVersion = buildToolsVersion;
    return this;
  }

  /** Sets the variant type to be used for {@link #whenVariantIs}. */
  public AaptCommandBuilder forVariantType(VariantType variantType) {
    Preconditions.checkNotNull(variantType);
    Preconditions.checkState(this.variantType == null, "A variant type was already specified.");
    this.variantType = variantType;
    return this;
  }

  /** Adds a single flag to the builder. */
  public AaptCommandBuilder add(String flag) {
    flags.add(flag);
    return this;
  }

  /**
   * Adds a flag to the builder, along with a string value. The two will be added as different words
   * in the final command line. If the value is {@code null}, neither the flag nor the value will be
   * added.
   */
  public AaptCommandBuilder add(String flag, @Nullable String value) {
    Preconditions.checkNotNull(flag);
    if (!Strings.isNullOrEmpty(value)) {
      flags.add(flag);
      flags.add(value);
    }
    return this;
  }

  /**
   * Adds a flag to the builder, along with a path value. The path will be converted to a string
   * using {@code toString}, then the flag and the path will be added to the final command line as
   * different words. If the value is {@code null}, neither the flag nor the path will be added.
   *
   * @see #add(String,String)
   */
  public AaptCommandBuilder add(String flag, @Nullable Path path) {
    Preconditions.checkNotNull(flag);
    if (path != null) {
      add(flag, path.toString());
    }
    return this;
  }

  /**
   * Adds a flag to the builder multiple times, once for each value in the given collection. {@code
   * null} values will be skipped. If the collection is empty, nothing will be added. The values
   * will be added in the source collection's iteration order.
   *
   * <p>ex. If {@code flag} is {@code "-0"} and {@code values} contains the values {@code "png"},
   * {@code null}, and {@code "gif"}, then four words will be added to the final command line:
   * {@code "-0", "png", "-0", "gif"}.
   */
  public AaptCommandBuilder addRepeated(String flag, Collection<String> values) {
    Preconditions.checkNotNull(flag);
    for (String value : values) {
      add(flag, value);
    }
    return this;
  }

  /**
   * Adds a flag to the builder multiple times, once for each value in the given collection. {@code
   * null} values will be skipped. If the collection is empty, nothing will be added. The values
   * will be added in the source collection's iteration order. See {@link
   * AaptCommandBuilder#addRepeated(String, Collection)} for more information. If the collection
   * exceed 200 items, the values will be written to a file and passed as &lt;flag&gt @&lt;file&gt;.
   */
  public AaptCommandBuilder addParameterableRepeated(
      final String flag, Collection<String> values, Path workingDirectory) throws IOException {
    Preconditions.checkNotNull(flag);
    Preconditions.checkNotNull(workingDirectory);
    if (values.size() > 200) {
      add(
          flag,
          "@"
              + Files.write(
                  Files.createDirectories(workingDirectory).resolve("params" + flag),
                  ImmutableList.of(values.stream().collect(Collectors.joining(" ")))));
    } else {
      addRepeated(flag, values);
    }
    return this;
  }

  /** Adds the next flag to the builder only if the condition is true. */
  public ConditionalAaptCommandBuilder when(boolean condition) {
    if (condition) {
      return new SuccessfulConditionCommandBuilder(this);
    } else {
      return new FailedConditionCommandBuilder(this);
    }
  }

  /** Adds the next flag to the builder only if the variant type is the passed-in type. */
  public ConditionalAaptCommandBuilder whenVariantIs(VariantType variantType) {
    Preconditions.checkNotNull(variantType);
    return when(this.variantType == variantType);
  }

  /**
   * Adds the next flag to the builder only if the build tools version is unspecified or is greater
   * than or equal to the given version.
   */
  public ConditionalAaptCommandBuilder whenVersionIsAtLeast(Revision requiredVersion) {
    Preconditions.checkNotNull(requiredVersion);
    return when(buildToolsVersion == null || buildToolsVersion.compareTo(requiredVersion) >= 0);
  }

  /** Assembles the full command line as a list. */
  public List<String> build() {
    return flags.build();
  }

  public AaptCommandBuilder add(String flag, Optional<Path> optionalPath) {
    Preconditions.checkNotNull(flag);
    Preconditions.checkNotNull(optionalPath);
    optionalPath.ifPresent(p -> add(flag, p));
    return this;
  }

  /** Wrapper for potentially adding flags to an AaptCommandBuilder based on a conditional. */
  public interface ConditionalAaptCommandBuilder {
    /**
     * Adds a single flag to the builder if the condition was true.
     *
     * @see AaptCommandBuilder#add(String)
     */
    AaptCommandBuilder thenAdd(String flag);

    /**
     * Adds a single flag and associated string value to the builder if the value is non-null and
     * the condition was true.
     *
     * @see AaptCommandBuilder#add(String,String)
     */
    AaptCommandBuilder thenAdd(String flag, @Nullable String value);

    /**
     * Adds a single flag and associated path value to the builder if the value is non-null and the
     * condition was true.
     *
     * @see AaptCommandBuilder#add(String,Path)
     */
    AaptCommandBuilder thenAdd(String flag, @Nullable Path value);

    /**
     * Adds a single flag and associated path value to the builder if the value is non-null and the
     * condition was true.
     *
     * @see AaptCommandBuilder#add(String,Optional)
     */
    AaptCommandBuilder thenAdd(String flag, Optional<Path> value);

    /**
     * Adds the values in the collection to the builder, each preceded by the given flag, if the
     * collection was non-empty and the condition was true.
     *
     * @see AaptCommandBuilder#addRepeated(String,Collection<String>)
     */
    AaptCommandBuilder thenAddRepeated(String flag, Collection<String> values);
  }

  /**
   * Forwarding implementation of ConditionalAaptCommandBuilder returned when a condition is true.
   */
  private static class SuccessfulConditionCommandBuilder implements ConditionalAaptCommandBuilder {
    private final AaptCommandBuilder originalCommandBuilder;

    public SuccessfulConditionCommandBuilder(AaptCommandBuilder originalCommandBuilder) {
      this.originalCommandBuilder = originalCommandBuilder;
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag) {
      return originalCommandBuilder.add(flag);
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag, @Nullable String value) {
      return originalCommandBuilder.add(flag, value);
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag, @Nullable Path value) {
      return originalCommandBuilder.add(flag, value);
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag, @Nullable Optional<Path> value) {
      return originalCommandBuilder.add(flag, value);
    }

    @Override
    public AaptCommandBuilder thenAddRepeated(String flag, Collection<String> values) {
      return originalCommandBuilder.addRepeated(flag, values);
    }
  }

  /** Null implementation of ConditionalAaptCommandBuilder returned when a condition is false. */
  private static class FailedConditionCommandBuilder implements ConditionalAaptCommandBuilder {
    private final AaptCommandBuilder originalCommandBuilder;

    public FailedConditionCommandBuilder(AaptCommandBuilder originalCommandBuilder) {
      this.originalCommandBuilder = originalCommandBuilder;
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag) {
      Preconditions.checkNotNull(flag);
      return originalCommandBuilder;
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag, @Nullable String value) {
      Preconditions.checkNotNull(flag);
      return originalCommandBuilder;
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag, @Nullable Path value) {
      Preconditions.checkNotNull(flag);
      return originalCommandBuilder;
    }

    @Override
    public AaptCommandBuilder thenAdd(String flag, Optional<Path> value) {
      Preconditions.checkNotNull(flag);
      Preconditions.checkNotNull(value);
      return originalCommandBuilder;
    }

    @Override
    public AaptCommandBuilder thenAddRepeated(String flag, Collection<String> values) {
      Preconditions.checkNotNull(flag);
      Preconditions.checkNotNull(values);
      return originalCommandBuilder;
    }
  }

  /**
   * Executes command and returns log.
   *
   * @throws IOException when the process cannot execute.
   */
  public String execute(String action) throws IOException {
    return CommandHelper.execute(action, build());
  }
}
