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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;

/** Action that writes a native {@code .exe} launcher for {java,sh,py}_binary rules on Windows. */
public final class LauncherFileWriteAction extends AbstractFileWriteAction {

  // Generated with /usr/bin/uuidgen.
  // This GUID doesn't have to be anything specific, we only use it to salt action cache keys so it
  // just has to be unique among other actions.
  private static final String GUID = "1f57afe7-f6f8-487c-9a8a-0a0286172fef";

  private final LaunchInfo launchInfo;
  private final Artifact launcher;
  private final boolean isExecutedOnWindows;

  /** Creates a new {@link LauncherFileWriteAction}, registering it with the {@code ruleContext}. */
  public static void createAndRegister(
      RuleContext ruleContext, Artifact output, LaunchInfo launchInfo) {
    ruleContext.registerAction(
        new LauncherFileWriteAction(
            ruleContext, output, ruleContext.getPrerequisiteArtifact("$launcher"), launchInfo));
  }

  /** Creates a new {@code LauncherFileWriteAction}. */
  private LauncherFileWriteAction(
      RuleContext ruleContext, Artifact output, Artifact launcher, LaunchInfo launchInfo) {
    super(
        ruleContext.getActionOwner(),
        NestedSetBuilder.create(Order.STABLE_ORDER, Preconditions.checkNotNull(launcher)),
        output,
        /* makeExecutable= */ true);
    this.launcher = launcher; // already null-checked in the superclass c'tor
    this.launchInfo = Preconditions.checkNotNull(launchInfo);
    this.isExecutedOnWindows = ruleContext.isExecutedOnWindows();
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    // TODO(tjgq): Move this check into createAndRegister.
    // This requires fixing many unit tests that don't appropriately set the execution platform
    // when running on Windows.
    checkState(isExecutedOnWindows);
    return out -> {
      try (InputStream in = ctx.getInputPath(this.launcher).getInputStream()) {
        ByteStreams.copy(in, out);
      }
      long dataLength = this.launchInfo.write(out);
      ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
      buffer.order(ByteOrder.LITTLE_ENDIAN); // All Windows versions are little endian.
      buffer.putLong(dataLength);
      out.write(buffer.array());
      out.flush();
    };
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    fp.addPath(this.launcher.getExecPath());
    fp.addString(this.launchInfo.fingerPrint);
  }

  /**
   * Metadata that describes the payload of the native launcher binary.
   *
   * <p>This object constructs the binary metadata lazily, to save memory.
   */
  public static final class LaunchInfo {

    /** Precomputed fingerprint of this object. */
    public final String fingerPrint;

    private final ImmutableList<Entry> entries;

    private LaunchInfo(ImmutableList<Entry> entries) {
      this.entries = entries;
      this.fingerPrint = computeKey(entries);
    }

    /** Creates a new {@link Builder}. */
    public static Builder builder() {
      return new Builder();
    }

    /** Writes this object's entries to {@code out}, returns the total written amount in bytes. */
    @VisibleForTesting
    long write(OutputStream out) throws IOException {
      long len = 0;
      for (Entry e : entries) {
        len += e.write(out);
        out.write('\0');
        ++len;
      }
      return len;
    }

    /** Computes the fingerprint of the {@code entries}. */
    private static String computeKey(ImmutableList<Entry> entries) {
      Fingerprint f = new Fingerprint();
      for (Entry e : entries) {
        e.addToFingerprint(f);
      }
      return f.hexDigestAndReset();
    }

    /** Writes {@code s} to {@code out} encoded as UTF-8, returns the written length in bytes. */
    private static long writeString(String s, OutputStream out) throws IOException {
      byte[] b = s.getBytes(StandardCharsets.UTF_8);
      out.write(b);
      return b.length;
    }

    /** Represents one entry in {@link LaunchInfo.entries}. */
    private static interface Entry {
      /** Writes this entry to {@code out}, returns the written length in bytes. */
      long write(OutputStream out) throws IOException;

      /** Adds this entry to the fingerprint computer {@code f}. */
      void addToFingerprint(Fingerprint f);
    }

    /** A key-value pair entry. */
    private static final class KeyValuePair implements Entry {
      private final String key;
      @Nullable private final String value;

      public KeyValuePair(String key, @Nullable String value) {
        this.key = Preconditions.checkNotNull(key);
        this.value = value;
      }

      @Override
      public long write(OutputStream out) throws IOException {
        long len = writeString(key, out);
        len += writeString("=", out);
        if (value != null && !value.isEmpty()) {
          len += writeString(value, out);
        }
        return len;
      }

      @Override
      public void addToFingerprint(Fingerprint f) {
        f.addString(key);
        f.addString(value != null ? value : "");
      }
    }

    /** A pair of a key and a delimiter-joined list of values. */
    private static final class JoinedValues implements Entry {
      private final String key;
      private final String delimiter;
      @Nullable private final Iterable<String> values;

      public JoinedValues(String key, String delimiter, @Nullable Iterable<String> values) {
        this.key = Preconditions.checkNotNull(key);
        this.delimiter = Preconditions.checkNotNull(delimiter);
        this.values = values;
      }

      @Override
      public long write(OutputStream out) throws IOException {
        long len = writeString(key, out);
        len += writeString("=", out);
        if (values != null) {
          boolean first = true;
          for (String v : values) {
            if (first) {
              first = false;
            } else {
              len += writeString(delimiter, out);
            }
            len += writeString(v, out);
          }
        }
        return len;
      }

      @Override
      public void addToFingerprint(Fingerprint f) {
        f.addString(key);
        if (values != null) {
          for (String v : values) {
            f.addString(v != null ? v : "");
          }
        }
      }
    }

    /** Builder for {@link LaunchInfo} instances. */
    public static final class Builder {
      private ImmutableList.Builder<Entry> entries = ImmutableList.builder();

      /** Builds a {@link LaunchInfo} from this builder. This builder may be reused. */
      public LaunchInfo build() {
        return new LaunchInfo(entries.build());
      }

      /**
       * Adds a key-value pair entry.
       *
       * <p>Examples:
       *
       * <ul>
       *   <li>{@code key} is "foo" and {@code value} is "bar", the written value is "foo=bar\0"
       *   <li>{@code key} is "foo" and {@code value} is null or empty, the written value is
       *       "foo=\0"
       * </ul>
       */
      @CanIgnoreReturnValue
      public Builder addKeyValuePair(String key, @Nullable String value) {
        Preconditions.checkNotNull(key);
        if (!key.isEmpty()) {
          entries.add(new KeyValuePair(key, value));
        }
        return this;
      }

      /**
       * Adds a key and list of lazily-joined values.
       *
       * <p>Examples:
       *
       * <ul>
       *   <li>{@code key} is "foo", {@code delimiter} is ";", {@code values} is ["bar", "baz",
       *       "qux"], the written value is "foo=bar;baz;qux\0"
       *   <li>{@code key} is "foo", {@code delimiter} is irrelevant, {@code value} is null or
       *       empty, the written value is "foo=\0"
       * </ul>
       */
      @CanIgnoreReturnValue
      public Builder addJoinedValues(
          String key, String delimiter, @Nullable Iterable<String> values) {
        Preconditions.checkNotNull(key);
        Preconditions.checkNotNull(delimiter);
        if (!key.isEmpty()) {
          entries.add(new JoinedValues(key, delimiter, values));
        }
        return this;
      }
    }
  }
}
