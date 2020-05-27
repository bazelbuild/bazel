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

package com.google.devtools.build.lib.analysis.actions;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LazyString;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Action to write a file whose contents are known at analysis time.
 *
 * <p>The output is always UTF-8 encoded.
 *
 * <p>TODO(bazel-team): Choose a better name to distinguish this class from {@link
 * BinaryFileWriteAction}.
 */
@AutoCodec
@Immutable // if fileContents is immutable
public final class FileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "332877c7-ca9f-4731-b387-54f620408522";

  /**
   * The contents may be lazily computed or compressed.
   *
   * <p>If the object representing the contents is a {@code String}, its length is greater than
   * {@code COMPRESS_CHARS_THRESHOLD}, and compression is enabled, then the gzipped bytestream of
   * the contents will be stored in place of the string itself. This compression is transparent and
   * does not affect the output file.
   *
   * <p>Otherwise, if the object represents a lazy computation, it will not be forced until {@link
   * #getFileContents()} is called. An example where this may come in handy is if the contents are
   * the concatenation of the string representations of a series of artifacts. Then the client code
   * can wrap a {@code List<Artifact>} in a {@code LazyString}, which saves memory since the
   * artifacts are shared objects whereas a string is not.
   */
  private final CharSequence fileContents;

  /** Minimum length (in chars) for content to be eligible for compression. */
  private static final int COMPRESS_CHARS_THRESHOLD = 256;

  private FileWriteAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Artifact output,
      CharSequence fileContents,
      boolean makeExecutable,
      Compression allowCompression) {
    this(
        owner,
        inputs,
        output,
        allowCompression == Compression.ALLOW
                && fileContents instanceof String
                && fileContents.length() > COMPRESS_CHARS_THRESHOLD
            ? new CompressedString((String) fileContents)
            : fileContents,
        makeExecutable);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  FileWriteAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Artifact primaryOutput,
      CharSequence fileContents,
      boolean makeExecutable) {
    super(owner, inputs, primaryOutput, makeExecutable);
    this.fileContents = fileContents;
  }

  /**
   * Creates a new FileWriteAction instance with inputs and empty content.
   *
   * <p>This is useful for producing an artifact that, if built, will ensure that the generating
   * actions for its inputs are run. The output file is non-executable.
   *
   * @param owner the action owner
   * @param inputs the Artifacts that this Action depends on
   * @param output the Artifact that will be created by executing this Action
   */
  public static FileWriteAction createEmptyWithInputs(
      ActionOwner owner, NestedSet<Artifact> inputs, Artifact output) {
    return new FileWriteAction(owner, inputs, output, "", false, Compression.DISALLOW);
  }

  /**
   * Creates a new FileWriteAction instance with direct control over whether or not transparent
   * compression may be used.
   *
   * @param owner the action owner
   * @param output the Artifact that will be created by executing this Action
   * @param fileContents the contents to be written to the file
   * @param makeExecutable whether the output file is made executable
   * @param allowCompression whether (transparent) compression is enabled
   */
  public static FileWriteAction create(
      ActionOwner owner,
      Artifact output,
      CharSequence fileContents,
      boolean makeExecutable,
      Compression allowCompression) {
    return new FileWriteAction(
        owner,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        output,
        fileContents,
        makeExecutable,
        allowCompression);
  }

  /**
   * Creates a new FileWriteAction instance.
   *
   * <p>There are no inputs. Transparent compression is controlled by the {@code
   * --experimental_transparent_compression} flag. No reference to the {@link
   * ActionConstructionContext} will be maintained.
   *
   * @param context the action construction context
   * @param output the Artifact that will be created by executing this Action
   * @param fileContents the contents to be written to the file
   * @param makeExecutable whether the output file is made executable
   */
  public static FileWriteAction create(
      ActionConstructionContext context,
      Artifact output,
      CharSequence fileContents,
      boolean makeExecutable) {
    return new FileWriteAction(
        context.getActionOwner(),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        output,
        fileContents,
        makeExecutable,
        context.getConfiguration().transparentCompression());
  }

  private static final class CompressedString extends LazyString {
    final byte[] bytes;
    final int uncompressedSize;

    CompressedString(String chars) {
      byte[] dataToCompress = chars.getBytes(UTF_8);
      ByteArrayOutputStream byteStream = new ByteArrayOutputStream(dataToCompress.length);
      try (GZIPOutputStream zipStream = new GZIPOutputStream(byteStream)) {
        zipStream.write(dataToCompress);
      } catch (IOException e) {
        // This should be impossible since we're writing to a byte array.
        throw new RuntimeException(e);
      }
      this.uncompressedSize = dataToCompress.length;
      this.bytes = byteStream.toByteArray();
    }

    @Override
    public String toString() {
      byte[] uncompressedBytes = new byte[uncompressedSize];
      try (GZIPInputStream zipStream = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
        int read;
        int totalRead = 0;
        while (totalRead < uncompressedSize
            && (read = zipStream.read(uncompressedBytes, totalRead, uncompressedSize - totalRead))
                != -1) {
          totalRead += read;
        }
        if (totalRead != uncompressedSize) {
          // This should be impossible.
          throw new RuntimeException("Corrupt byte buffer in FileWriteAction.");
        }
      } catch (IOException e) {
        // This should be impossible since we're reading from a byte array.
        throw new RuntimeException(e);
      }
      return new String(uncompressedBytes, UTF_8);
    }
  }

  @VisibleForTesting
  boolean usesCompression() {
    return fileContents instanceof CompressedString;
  }

  /**
   * Returns the string contents to be written.
   *
   * <p>Note that if the string is lazily computed or compressed, calling this method will force its
   * computation or decompression. No attempt is made by FileWriteAction to cache the result.
   */
  public String getFileContents() {
    return fileContents.toString();
  }

  @Override
  public String getStarlarkContent() {
    return getFileContents();
  }

  /**
   * Create a DeterministicWriter for the content of the output file as provided by
   * {@link #getFileContents()}.
   */
  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        out.write(getFileContents().getBytes(UTF_8));
      }
    };
  }

  /** Computes the Action key for this action by computing the fingerprint for the file contents. */
  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addString(String.valueOf(makeExecutable));
    fp.addString(getFileContents());
  }

  /**
   * Creates a FileWriteAction to write contents to the resulting artifact fileName in the genfiles
   * root underneath the package path.
   *
   * @param ruleContext the ruleContext that will own the action of creating this file
   * @param fileName name of the file to create
   * @param contents data to write to file
   * @param executable flags that file should be marked executable
   * @return Artifact describing the file to create
   */
  public static Artifact createFile(
      RuleContext ruleContext, String fileName, CharSequence contents, boolean executable) {
    Artifact scriptFileArtifact = ruleContext.getPackageRelativeArtifact(
        fileName, ruleContext.getConfiguration().getGenfilesDirectory(
            ruleContext.getRule().getRepository()));
    ruleContext.registerAction(
        FileWriteAction.create(ruleContext, scriptFileArtifact, contents, executable));
    return scriptFileArtifact;
  }
}
