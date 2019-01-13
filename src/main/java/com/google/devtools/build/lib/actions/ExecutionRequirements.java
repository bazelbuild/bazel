// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.auto.value.AutoValue;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Strings used to express requirements on action execution environments.
 */
public class ExecutionRequirements {

  /** An execution requirement that can be split into a key and a value part using a regex. */
  @AutoValue
  public abstract static class ParseableRequirement {

    /**
     * Thrown when a {@link ParseableRequirement} feels responsible for a tag, but the {@link
     * #validator()} method returns an error.
     */
    public static class ValidationException extends Exception {
      private final String tagValue;

      /**
       * Creates a new {@link ValidationException}.
       *
       * @param tagValue the erroneous value that was parsed from the tag.
       * @param errorMsg an error message that tells the user what's wrong with the value.
       */
      public ValidationException(String tagValue, String errorMsg) {
        super(errorMsg);
        this.tagValue = tagValue;
      }

      /**
       * Returns the erroneous value of the parsed tag.
       *
       * <p>Useful to put in error messages shown to the user.
       */
      public String getTagValue() {
        return tagValue;
      }
    }

    /**
     * Create a new parseable execution requirement definition.
     *
     * <p>If a tag doesn't match the detectionPattern, it will be ignored. If a tag matches the
     * detectionPattern, but not the validationPattern, it is assumed that the value is somehow
     * wrong (e.g. the user put a float or random string where we expected an integer).
     *
     * @param userFriendlyName a human readable name of the tag and its format, e.g. "cpu:<int>"
     * @param detectionPattern a regex that will be used to detect whether a tag matches this
     *     execution requirement. It should have one capture group that grabs the value of the tag.
     *     This should be general enough to permit even wrong value types. Example: "cpu:(.+)".
     * @param validator a Function that will be used to validate the value of the tag. It should
     *     return null if the value is fine to use or a human-friendly error message describing why
     *     the value is not valid.
     */
    static ParseableRequirement create(
        String userFriendlyName, Pattern detectionPattern, Function<String, String> validator) {
      return new AutoValue_ExecutionRequirements_ParseableRequirement(
          userFriendlyName, detectionPattern, validator);
    }

    public abstract String userFriendlyName();

    public abstract Pattern detectionPattern();

    public abstract Function<String, String> validator();

    /**
     * Returns the parsed value from a tag, if this {@link ParseableRequirement} detects that it is
     * responsible for it, otherwise returns {@code null}.
     *
     * @throws ValidationException if the value parsed out of the tag doesn't pass the validator.
     */
    public String parseIfMatches(String tag) throws ValidationException {
      Matcher matcher = detectionPattern().matcher(tag);
      if (!matcher.matches()) {
        return null;
      }
      String tagValue = matcher.group(1);
      String errorMsg = validator().apply(tagValue);
      if (errorMsg != null) {
        throw new ValidationException(tagValue, errorMsg);
      }
      return tagValue;
    }
  }

  /** An execution requirement that is split into a key and a parseable value. */
  @AutoValue
  public abstract static class TaggedRequirement<ValueType> {
    /**
     * Thrown when the {@link #validator()} method returns an error.
     */
    public static class ValidationException extends Exception {
      private final String tagValue;

      /**
       * Creates a new {@link ValidationException}.
       *
       * @param tagValue the erroneous value that was parsed from the tag
       * @param errorMsg an error message that tells the user what's wrong with the value
       */
      public ValidationException(String tagValue, String errorMsg) {
        super(errorMsg);
        this.tagValue = tagValue;
      }

      /**
       * Returns the erroneous value of the parsed tag.
       *
       * <p>Useful to put in error messages shown to the user.
       */
      public String getTagValue() {
        return tagValue;
      }
    }

    /**
     * Create a new parseable execution requirement definition.
     *
     * @param tagName a human readable name of the tag
     * @param validator a Function that will be used to validate the value of the tag. It should
     *     return null if the value is fine to use or a human-friendly error message describing why
     *     the value is not valid.
     * @param parser a Function that will transform the value of the tag to {@link ValueType}.
     */
    static <ValueType> TaggedRequirement<ValueType> create(
        String tagName,
        Function<String, String> validator,
        Function<String, ValueType> parser) {
      return new AutoValue_ExecutionRequirements_TaggedRequirement<ValueType>(
          tagName, validator, parser);
    }

    public abstract String tagName();

    public abstract Function<String, String> validator();

    public abstract Function<String, ValueType> parser();

    /**
     * Returns the parsed value from a tag's value.
     *
     * @throws ValidationException if the value parsed out of the tag doesn't pass the validator.
     */
    public ValueType parse(String value) throws ValidationException {
      String errorMsg = validator().apply(value);
      if (errorMsg != null) {
        throw new ValidationException(value, errorMsg);
      }
      return parser().apply(value);
    }
  }

  /** Integer validator to use with {@link ParseableRequirement} and {@link TaggedRequirement}. */
  private static String IntegerValidator(String s) {
    Preconditions.checkNotNull(s);

    int value;
    try {
      value = Integer.parseInt(s);
    } catch (NumberFormatException e) {
      return "can't be parsed as an integer";
    }

    // De-and-reserialize & compare to only allow canonical integer formats.
    if (!Integer.toString(value).equals(s)) {
      return "must be in canonical format (e.g. '4' instead of '+04')";
    }

    if (value < 1) {
      return "can't be zero or negative";
    }

    return null;
  }

  /** If specified, the timeout of this action in seconds. Must be decimal integer. */
  public static final String TIMEOUT = "timeout";

  /** If an action would not successfully run other than on Darwin. */
  public static final String REQUIRES_DARWIN = "requires-darwin";

  /** Whether we should disable prefetching of inputs before running a local action. */
  public static final String DISABLE_LOCAL_PREFETCH = "disable-local-prefetch";

  /** How many hardware threads an action requires for execution. */
  public static final ParseableRequirement CPU =
      ParseableRequirement.create(
          "cpu:<int>",
          Pattern.compile("cpu:(.+)"),
          ExecutionRequirements::IntegerValidator);

  /** How many hardware threads an action requires for execution. */
  public static final TaggedRequirement<Integer> TAGGED_CPU =
      TaggedRequirement.create(
          "cpu",
          ExecutionRequirements::IntegerValidator,
          Integer::parseInt);

  private static final Pattern MEMORY_PATTERN = Pattern.compile("^(\\d+(\\.\\d+)?)([GM])$");

  /**
   * How much real memory (resident set size) this action requires for execution.
   *
   * Input must be an integer or trivial floating point (no sign or exponent)
   * followed by an "M" or "G" suffix to indicate megabytes or gigabytes,
   * respectively.
   */
  public static final TaggedRequirement<Integer> TAGGED_MEM_MB =
      TaggedRequirement.create(
          "memory",
          s -> {
            Matcher m = MEMORY_PATTERN.matcher(s);
            if (!m.matches()) {
              return String.format("must match pattern '%s'", MEMORY_PATTERN.pattern());
            }
            try {
              float value = Float.parseFloat(m.group(1));
            } catch (NumberFormatException e) {
              return "can't be parsed as a float";
            }
            return null;
          },
          s -> {
            Matcher m = MEMORY_PATTERN.matcher(s);
            m.matches();  // Required before m.group; guaranteed per validator above.
            float memoryMb = Float.parseFloat(m.group(1));
            if (m.group(3).equals("G")) {
              memoryMb *= 1024;
            }
            return (int)memoryMb;
          });

  /** If an action supports running in persistent worker mode. */
  public static final String SUPPORTS_WORKERS = "supports-workers";

  public static final ImmutableMap<String, String> WORKER_MODE_ENABLED =
      ImmutableMap.of(SUPPORTS_WORKERS, "1");

  /**
   * Requires local execution without sandboxing for a spawn.
   *
   * <p>This tag is deprecated; use no-cache, no-remote, or no-sandbox instead.
   */
  public static final String LOCAL = "local";

  /**
   * Disables local and remote caching for a spawn, but note that the local action cache may still
   * apply.
   *
   * <p>This tag can also be set on an action, in which case it completely disables all caching for
   * that action, but note that action-generated spawns may still be cached, unless they also carry
   * this tag.
   */
  public static final String NO_CACHE = "no-cache";

  /** Disables local sandboxing of a spawn. */
  public static final String LEGACY_NOSANDBOX = "nosandbox";

  /** Disables local sandboxing of a spawn. */
  public static final String NO_SANDBOX = "no-sandbox";

  /** Disables remote execution of a spawn. */
  public static final String NO_REMOTE = "no-remote";

  /**
   * Enables networking for a spawn if possible (only if sandboxing is enabled and if the sandbox
   * supports it).
   */
  public static final String REQUIRES_NETWORK = "requires-network";

  /**
   * Disables networking for a spawn if possible (only if sandboxing is enabled and if the sandbox
   * supports it).
   */
  public static final String BLOCK_NETWORK = "block-network";

  /**
   * On linux, if sandboxing is enabled, ensures that a spawn is run with uid 0, i.e., root. Has no
   * effect otherwise.
   */
  public static final String REQUIRES_FAKEROOT = "requires-fakeroot";

  /** Suppress CLI reporting for this spawn - it's part of another action. */
  public static final String DO_NOT_REPORT = "internal-do-not-report";

  /** Use this to request eager fetching of a single remote output into local memory. */
  public static final String REMOTE_EXECUTION_INLINE_OUTPUTS = "internal-inline-outputs";
}
