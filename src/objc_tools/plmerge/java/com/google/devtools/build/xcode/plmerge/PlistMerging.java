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

package com.google.devtools.build.xcode.plmerge;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.Sets;
import com.google.common.io.ByteSource;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos.Control;
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Mapping;
import com.google.devtools.build.xcode.util.Value;

import com.dd.plist.BinaryPropertyListWriter;
import com.dd.plist.NSDictionary;
import com.dd.plist.NSObject;
import com.dd.plist.NSString;
import com.dd.plist.PropertyListFormatException;
import com.dd.plist.PropertyListParser;

import org.xml.sax.SAXException;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import javax.xml.parsers.ParserConfigurationException;

/**
 * Utility code for merging project files.
 */
public class PlistMerging extends Value<PlistMerging> {
  private static final String BUNDLE_IDENTIFIER_PLIST_KEY = "CFBundleIdentifier";
  private static final String BUNDLE_IDENTIFIER_DEFAULT = "com.generic.bundleidentifier";
  private static final String BUNDLE_VERSION_PLIST_KEY = "CFBundleVersion";
  private static final String BUNDLE_VERSION_DEFAULT = "1.0.0";
  private static final String BUNDLE_SHORT_VERSION_STRING_PLIST_KEY = "CFBundleShortVersionString";
  private static final String BUNDLE_SHORT_VERSION_STRING_DEFAULT = "1.0";

  /**
   * Exception type thrown when validation of the plist file fails.
   */
  public static class ValidationException extends RuntimeException {
    ValidationException(String message) {
      super(message);
    }
  }

  private final NSDictionary merged;

  /**
   * Wraps a {@code NSDictionary} as a PlistMerging.
   */
  public PlistMerging(NSDictionary merged) {
    super(merged);
    this.merged = merged;
  }

  /**
   * Merges several plist files into a single {@code NSDictionary}. Each file should be a plist (of
   * one of these formats: ASCII, Binary, or XML) that contains an NSDictionary.
   */
  @VisibleForTesting
  static NSDictionary merge(Iterable<? extends Path> sourceFilePaths) throws IOException {
    NSDictionary result = new NSDictionary();
    for (Path sourceFilePath : sourceFilePaths) {
      result.putAll(readPlistFile(sourceFilePath));
    }
    return result;
  }

  public static NSDictionary readPlistFile(final Path sourceFilePath) throws IOException {
    ByteSource rawBytes = new Utf8BomSkippingByteSource(sourceFilePath);

    try {
      try (InputStream in = rawBytes.openStream()) {
        return (NSDictionary) PropertyListParser.parse(in);
      } catch (PropertyListFormatException | ParseException e) {
        // If we failed to parse, the plist may implicitly be a map. To handle this, wrap the plist
        // with {}.
        // TODO(bazel-team): Do this in a cleaner way.
        ByteSource concatenated = ByteSource.concat(
            ByteSource.wrap(new byte[] {'{'}),
            rawBytes,
            ByteSource.wrap(new byte[] {'}'}));
        try (InputStream in = concatenated.openStream()) {
          return (NSDictionary) PropertyListParser.parse(in);
        }
      }
    } catch (PropertyListFormatException | ParseException | ParserConfigurationException
        | SAXException e) {
      throw new IOException(e);
    }
  }

  /**
   * Writes the results of a merge operation to a plist file.
   * @param plistPath the path of the plist to write in binary format
   */
  public PlistMerging writePlist(Path plistPath) throws IOException {
    try (OutputStream out = Files.newOutputStream(plistPath)) {
      BinaryPropertyListWriter.write(out, merged);
    }
    return this;
  }

  /**
   * Writes a PkgInfo file based on certain keys in the merged plist.
   * @param pkgInfoPath the path of the PkgInfo file to write. In many iOS apps, this file just
   *     contains the raw string {@code APPL????}.
   */
  public PlistMerging writePkgInfo(Path pkgInfoPath) throws IOException {
    String pkgInfo =
        Mapping.of(merged, "CFBundlePackageType").or(NSObject.wrap("APPL")).toString()
        + Mapping.of(merged, "CFBundleSignature").or(NSObject.wrap("????")).toString();
    Files.write(pkgInfoPath, pkgInfo.getBytes(StandardCharsets.UTF_8));
    return this;
  }

  /**
   * Generates a Plistmerging combining values from sourceFiles and immutableSourceFiles, and
   * modifying them based on substitutions and keysToRemoveIfEmptyString.
   */
  public static PlistMerging from(
      Control control,
      KeysToRemoveIfEmptyString keysToRemoveIfEmptyString)
      throws IOException {

    FileSystem fileSystem = FileSystems.getDefault();

    ImmutableList.Builder<Path> sourceFilePathsBuilder = new Builder<>();
    for (String pathString : control.getSourceFileList()) {
      sourceFilePathsBuilder.add(fileSystem.getPath(pathString));
    }
    ImmutableList.Builder<Path> immutableSourceFilePathsBuilder = new Builder<>();
    for (String pathString : control.getImmutableSourceFileList()) {
      immutableSourceFilePathsBuilder.add(fileSystem.getPath(pathString));
    }

    return from(
        sourceFilePathsBuilder.build(),
        immutableSourceFilePathsBuilder.build(),
        control.getVariableSubstitutionMap(),
        keysToRemoveIfEmptyString,
        Strings.emptyToNull(control.getExecutableName()));
  }

  /**
   * Generates a Plistmerging combining values from sourceFiles and immutableSourceFiles, and
   * modifying them based on subsitutions and keysToRemoveIfEmptyString.
   */
  public static PlistMerging from(
      List<Path> sourceFiles,
      List<Path> immutableSourceFiles,
      Map<String, String> substitutions,
      KeysToRemoveIfEmptyString keysToRemoveIfEmptyString,
      String executableName)
      throws IOException {
    NSDictionary merged = PlistMerging.merge(sourceFiles);
    NSDictionary immutableEntries = PlistMerging.merge(immutableSourceFiles);
    Set<String> conflictingEntries = Sets.intersection(immutableEntries.keySet(), merged.keySet());

    Preconditions.checkArgument(
        conflictingEntries.isEmpty(),
        "The following plist entries may not be overridden, but are present in more than one "
            + "of the input lists: %s",
        conflictingEntries);
    merged.putAll(immutableEntries);

    for (Map.Entry<String, NSObject> entry : merged.entrySet()) {
      if (entry.getValue().toJavaObject() instanceof String) {
        String newValue = substituteEnvironmentVariable(
            substitutions, (String) entry.getValue().toJavaObject());
        merged.put(entry.getKey(), newValue);
      }
    }

    for (String key : keysToRemoveIfEmptyString) {
      if (Equaling.of(Mapping.of(merged, key), Optional.<NSObject>of(new NSString("")))) {
        merged.remove(key);
      }
    }

    // Info.plist files must contain a valid CFBundleVersion and a valid CFBundleShortVersionString,
    // or it will be rejected by Apple.
    // A valid Bundle Version is 18 characters or less, and only contains [0-9.]
    // We know we have an info.plist file as opposed to a strings file if the immutableEntries
    // have any values set.
    // TODO(bazel-team): warn user if we replace their values.
    if (!immutableEntries.isEmpty()) {
      Pattern versionPattern = Pattern.compile("[^0-9.]");
      if (!merged.containsKey(BUNDLE_VERSION_PLIST_KEY)) {
        merged.put(BUNDLE_VERSION_PLIST_KEY, BUNDLE_VERSION_DEFAULT);
      } else {
        NSObject nsVersion = merged.get(BUNDLE_VERSION_PLIST_KEY);
        String version = (String) nsVersion.toJavaObject();
        if (version.length() > 18 || versionPattern.matcher(version).find()) {
          merged.put(BUNDLE_VERSION_PLIST_KEY, BUNDLE_VERSION_DEFAULT);
        }
      }
      if (!merged.containsKey(BUNDLE_SHORT_VERSION_STRING_PLIST_KEY)) {
        merged.put(BUNDLE_SHORT_VERSION_STRING_PLIST_KEY, BUNDLE_SHORT_VERSION_STRING_DEFAULT);
      } else {
        NSObject nsVersion = merged.get(BUNDLE_SHORT_VERSION_STRING_PLIST_KEY);
        String version = (String) nsVersion.toJavaObject();
        if (version.length() > 18 || versionPattern.matcher(version).find()) {
          merged.put(BUNDLE_SHORT_VERSION_STRING_PLIST_KEY, BUNDLE_SHORT_VERSION_STRING_DEFAULT);
        }
      }
    }
    
    PlistMerging result = new PlistMerging(merged);

    if (executableName != null) {
      result.setExecutableName(executableName);
    }

    return result;
  }

  private static String substituteEnvironmentVariable(
      Map<String, String> substitutions, String string) {
    // The substitution is *not* performed recursively.
    for (Map.Entry<String, String> variable : substitutions.entrySet()) {
      String key = variable.getKey();
      String value = variable.getValue();
      string = string
          .replace("${" + key + "}", value)
          .replace("$(" + key + ")", value);
      key = key + ":rfc1034identifier";
      value = convertToRFC1034(value);
      string = string
          .replace("${" + key + "}", value)
          .replace("$(" + key + ")", value);
    }

    return string;
  }

  // Force RFC1034 compliance by changing any "bad" character to a '-'
  // This is essentially equivalent to what Xcode does.
  private static String convertToRFC1034(String value) {
    return value.replaceAll("[^-0-9A-Za-z.]", "-");
  }

  @VisibleForTesting
  NSDictionary asDictionary() {
    return merged;
  }

  /**
   * Sets the given executable name on this merged plist in the {@code CFBundleExecutable}
   * attribute.
   *
   * @param executableName name of the bundle executable
   * @return this plist merging
   * @throws ValidationException if the plist already contains an incompatible
   *    {@code CFBundleExecutable} entry
   */
  public PlistMerging setExecutableName(String executableName) {
    NSString bundleExecutable = (NSString) merged.get("CFBundleExecutable");

    if (bundleExecutable == null) {
      merged.put("CFBundleExecutable", executableName);
    } else if (!executableName.equals(bundleExecutable.getContent())) {
      throw new ValidationException(String.format(
          "Blaze generated the executable %s but the Plist CFBundleExecutable is %s",
          executableName, bundleExecutable));
    }

    return this;
  }
  
  /**
   * Sets the given identifier on this merged plist in the {@code CFBundleIdentifier}
   * attribute.
   *
   * @param primaryIdentifier used to set the bundle identifier or override the existing one from
   *     plist file, can be null
   * @param fallbackIdentifier used to set the bundle identifier if it is not set by plist file or
   *     primary identifier, can be null
   * @return this plist merging
   */
  public PlistMerging setBundleIdentifier(String primaryIdentifier, String fallbackIdentifier) {
    NSString bundleIdentifier = (NSString) merged.get(BUNDLE_IDENTIFIER_PLIST_KEY);

    if (primaryIdentifier != null) {
      merged.put(BUNDLE_IDENTIFIER_PLIST_KEY, convertToRFC1034(primaryIdentifier));
    } else if (bundleIdentifier == null) {
      if (fallbackIdentifier != null) {
        merged.put(BUNDLE_IDENTIFIER_PLIST_KEY, convertToRFC1034(fallbackIdentifier));
      } else {
        // TODO(bazel-team): We shouldn't be generating an info.plist in this case.
        merged.put(BUNDLE_IDENTIFIER_PLIST_KEY, BUNDLE_IDENTIFIER_DEFAULT);
      }
    }

    return this;
  }

  private static class Utf8BomSkippingByteSource extends ByteSource {

    private static final byte[] UTF8_BOM =
        new byte[] { (byte) 0xEF, (byte) 0xBB, (byte) 0xBF };

    private final Path path;

    public Utf8BomSkippingByteSource(Path path) {
      this.path = path;
    }

    @Override
    public InputStream openStream() throws IOException {
      InputStream stream = new BufferedInputStream(Files.newInputStream(path));
      stream.mark(UTF8_BOM.length);
      byte[] buffer = new byte[UTF8_BOM.length];
      int read = stream.read(buffer);
      stream.reset();

      if (UTF8_BOM.length == read && Arrays.equals(buffer, UTF8_BOM)) {
        stream.skip(UTF8_BOM.length);
      }

      return stream;
    }
  }
}
