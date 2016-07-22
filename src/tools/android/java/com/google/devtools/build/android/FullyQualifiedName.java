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

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.devtools.build.android.proto.SerializeFormat;

import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.ResourceQualifier;
import com.android.resources.ResourceType;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.CheckReturnValue;
import javax.annotation.concurrent.Immutable;

/**
 * Represents a fully qualified name for an android resource.
 *
 * Each resource name consists of the resource package, name, type, and qualifiers.
 */
@Immutable
public class FullyQualifiedName implements DataKey, Comparable<FullyQualifiedName> {
  public static final String DEFAULT_PACKAGE = "res-auto";
  private static final Joiner DASH_JOINER = Joiner.on('-');

  // To save on memory, always return one instance for each FullyQualifiedName.
  // Using a HashMap to deduplicate the instances -- the key retrieves a single instance.
  private static final ConcurrentMap<FullyQualifiedName, FullyQualifiedName> instanceCache =
      new ConcurrentHashMap<>();
  private static final AtomicInteger cacheHit = new AtomicInteger(0);

  /**
   * A factory for parsing an generating FullyQualified names with qualifiers and package.
   */
  public static class Factory {
    private static final String BCP_PREFIX = "b+";
    private static final Pattern PARSING_REGEX =
        Pattern.compile("(?:(?<package>[^:]+):){0,1}(?<type>[^-/]+)(?:[^/]*)/(?<name>.+)");
    public static final String INVALID_QUALIFIED_NAME_MESSAGE_NO_MATCH =
        String.format(
            "%%s is not a valid qualified name. "
                + "It should be in the pattern [package:]{%s}/resourcename",
            Joiner.on(",").join(ResourceType.values()));
    public static final String INVALID_QUALIFIED_NAME_MESSAGE_NO_TYPE_OR_NAME =
        String.format(
            "Could not find either resource type (%%s) or name (%%s) in %%s. "
                + "It should be in the pattern [package:]{%s}/resourcename",
            Joiner.on(",").join(ResourceType.values()));
    public static final String INVALID_QUALIFIERS = "%s contains invalid qualifiers.";
    private final ImmutableList<String> qualifiers;
    private final String pkg;

    private Factory(ImmutableList<String> qualifiers, String pkg) {
      this.qualifiers = qualifiers;
      this.pkg = pkg;
    }

    /** Creates a factory with default package from a directory name split on '-'. */
    public static Factory fromDirectoryName(String[] dirNameAndQualifiers) {
      return from(getQualifiers(dirNameAndQualifiers));
    }

    private static List<String> getQualifiers(String[] dirNameAndQualifiers) {
      PeekingIterator<String> rawQualifiers =
          Iterators.peekingIterator(Iterators.forArray(dirNameAndQualifiers));
      // Remove directory name
      rawQualifiers.next();
      List<String> unHandledLanguageRegionQualifiers = new ArrayList<>();
      List<String> unHandledDensityQualifiers = new ArrayList<>();
      List<String> unHandledUIModeQualifiers = new ArrayList<>();
      List<String> handledQualifiers = new ArrayList<>();
      // TODO(corysmith): Remove when FolderConfiguration is updated to handle BCP prefixes.
      // TODO(corysmith): Add back in handling for anydpi
      while (rawQualifiers.hasNext()) {
        String qualifier = rawQualifiers.next();
        if (qualifier.startsWith(BCP_PREFIX)) {
          // The b+local+script/region can't be handled.
          unHandledLanguageRegionQualifiers.add(qualifier);
        } else if ("es".equalsIgnoreCase(qualifier)
            && rawQualifiers.hasNext()
            && "419".equalsIgnoreCase(rawQualifiers.peek())) {
          // Replace the es-419.
          unHandledLanguageRegionQualifiers.add("b+es+419");
          // Consume the next value, as it's been replaced.
          rawQualifiers.next();
        } else if ("sr".equalsIgnoreCase(qualifier)
            && rawQualifiers.hasNext()
            && "rlatn".equalsIgnoreCase(rawQualifiers.peek())) {
          // Replace the sr-rLatn.
          unHandledLanguageRegionQualifiers.add("b+sr+Latn");
          // Consume the next value, as it's been replaced.
          rawQualifiers.next();
        } else if (qualifier.equals("watch")) {
          unHandledUIModeQualifiers.add(qualifier);
        } else {
          // This qualifier can probably be handled by FolderConfiguration.
          handledQualifiers.add(qualifier);
        }
      }
      // Create a configuration
      FolderConfiguration config = FolderConfiguration.getConfigFromQualifiers(handledQualifiers);
      // FolderConfiguration returns an unhelpful null when it considers the qualifiers to be
      // invalid.
      if (config == null) {
        throw new IllegalArgumentException(
            String.format(INVALID_QUALIFIERS, DASH_JOINER.join(dirNameAndQualifiers)));
      }
      config.normalize();

      // This is fragile but better than the Gradle scheme of just dropping
      // entire subtrees. 
      Builder<String> builder = ImmutableList.<String>builder();
      addIfNotNull(config.getCountryCodeQualifier(), builder);
      addIfNotNull(config.getNetworkCodeQualifier(), builder);
      if (unHandledLanguageRegionQualifiers.isEmpty()) {
        addIfNotNull(config.getLanguageQualifier(), builder);
        addIfNotNull(config.getRegionQualifier(), builder);
      } else {
        builder.addAll(unHandledLanguageRegionQualifiers);
      }
      addIfNotNull(config.getLayoutDirectionQualifier(), builder);
      addIfNotNull(config.getSmallestScreenWidthQualifier(), builder);
      addIfNotNull(config.getScreenWidthQualifier(), builder);
      addIfNotNull(config.getScreenHeightQualifier(), builder);
      addIfNotNull(config.getScreenSizeQualifier(), builder);
      addIfNotNull(config.getScreenRatioQualifier(), builder);
      addIfNotNullAndExist(config, "getScreenRoundQualifier", builder);
      addIfNotNull(config.getScreenOrientationQualifier(), builder);
      if (unHandledUIModeQualifiers.isEmpty()) {
        addIfNotNull(config.getUiModeQualifier(), builder);
      } else {
        builder.addAll(unHandledUIModeQualifiers);
      }
      addIfNotNull(config.getNightModeQualifier(), builder);
      if (unHandledDensityQualifiers.isEmpty()) {
        addIfNotNull(config.getDensityQualifier(), builder);
      } else {
        builder.addAll(unHandledDensityQualifiers);
      }
      addIfNotNull(config.getTouchTypeQualifier(), builder);
      addIfNotNull(config.getKeyboardStateQualifier(), builder);
      addIfNotNull(config.getTextInputMethodQualifier(), builder);
      addIfNotNull(config.getNavigationStateQualifier(), builder);
      addIfNotNull(config.getNavigationMethodQualifier(), builder);
      addIfNotNull(config.getScreenDimensionQualifier(), builder);
      addIfNotNull(config.getVersionQualifier(), builder);

      return builder.build();
    }

    private static void addIfNotNullAndExist(
        FolderConfiguration config, String methodName, Builder<String> builder) {
      try {
        Method method = config.getClass().getMethod(methodName);
        ResourceQualifier qualifier = (ResourceQualifier) method.invoke(config);
        if (qualifier != null) {
          builder.add(qualifier.getFolderSegment());
        }
      } catch (NoSuchMethodException
          | IllegalAccessException
          | IllegalArgumentException
          | InvocationTargetException e) {
        // Suppress the error and continue.
        return;
      }
    }

    private static void addIfNotNull(
        ResourceQualifier qualifier, ImmutableList.Builder<String> builder) {
      if (qualifier != null) {
        builder.add(qualifier.getFolderSegment());
      }
    }

    public static Factory from(List<String> qualifiers, String pkg) {
      return new Factory(ImmutableList.copyOf(qualifiers), pkg);
    }

    public static Factory from(List<String> qualifiers) {
      return from(ImmutableList.copyOf(qualifiers), DEFAULT_PACKAGE);
    }

    public FullyQualifiedName create(ResourceType resourceType, String resourceName) {
      return create(resourceType, resourceName, pkg);
    }

    public FullyQualifiedName create(ResourceType resourceType, String resourceName, String pkg) {
      return FullyQualifiedName.of(pkg, qualifiers, resourceType, resourceName);
    }

    /**
     * Parses a FullyQualifiedName from a string.
     *
     * @param raw A string in the expected format from
     *     [&lt;package&gt;:]&lt;ResourceType.name&gt;/&lt;resource name&gt;.
     * @throws IllegalArgumentException when the raw string is not valid qualified name.
     */
    public FullyQualifiedName parse(String raw) {
      Matcher matcher = PARSING_REGEX.matcher(raw);
      if (!matcher.matches()) {
        throw new IllegalArgumentException(
            String.format(INVALID_QUALIFIED_NAME_MESSAGE_NO_MATCH, raw));
      }
      String parsedPackage = matcher.group("package");
      ResourceType resourceType = ResourceType.getEnum(matcher.group("type"));
      String resourceName = matcher.group("name");

      if (resourceType == null || resourceName == null) {
        throw new IllegalArgumentException(
            String.format(
                INVALID_QUALIFIED_NAME_MESSAGE_NO_TYPE_OR_NAME, resourceType, resourceName, raw));
      }
      return FullyQualifiedName.of(
          parsedPackage == null ? pkg : parsedPackage, qualifiers, resourceType, resourceName);
    }

    /**
     * Generates a FullyQualifiedName for a file-based resource given the source Path.
     *
     * @param sourcePath the path of the file-based resource.
     * @throws IllegalArgumentException if the file-based resource has an invalid filename
     */
    public FullyQualifiedName parse(Path sourcePath) {
      return parse(deriveRawFullyQualifiedName(sourcePath));
    }

    private static String deriveRawFullyQualifiedName(Path source) {
      if (source.getNameCount() < 2) {
        throw new IllegalArgumentException(
            String.format(
                "The resource path %s is too short. "
                    + "The path is expected to be <resource type>/<file name>.",
                source));
      }
      String pathWithExtension =
          source.subpath(source.getNameCount() - 2, source.getNameCount()).toString();
      int extensionStart = pathWithExtension.indexOf('.');
      if (extensionStart > 0) {
        return pathWithExtension.substring(0, extensionStart);
      }
      return pathWithExtension;
    }

    // Grabs the extension portion of the path removed by deriveRawFullyQualifiedName.
    private static String getSourceExtension(Path source) {
      // TODO(corysmith): Find out if there is a filename parser utility.
      String fileName = source.getFileName().toString();
      int extensionStart = fileName.indexOf('.');
      if (extensionStart > 0) {
        return fileName.substring(extensionStart);
      }
      return "";
    }
  }

  public static boolean isOverwritable(FullyQualifiedName name) {
    return !(name.resourceType == ResourceType.ID || name.resourceType == ResourceType.STYLEABLE);
  }

  /**
   * Creates a new FullyQualifiedName with sorted qualifiers.
   *
   * @param pkg The resource package of the name. If unknown the default should be "res-auto"
   * @param qualifiers The resource qualifiers of the name, such as "en" or "xhdpi".
   * @param resourceType The resource type of the name.
   * @param resourceName The resource name of the name.
   * @return A new FullyQualifiedName.
   */
  public static FullyQualifiedName of(
      String pkg, List<String> qualifiers, ResourceType resourceType, String resourceName) {
    ImmutableList<String> immutableQualifiers = ImmutableList.copyOf(qualifiers);
    // TODO(corysmith): Address the GC thrash this creates by managing a simplified, mutable key to
    // do the instance check.
    FullyQualifiedName name =
        new FullyQualifiedName(pkg, immutableQualifiers, resourceType, resourceName);
    // Use putIfAbsent to get the canonical instance, if there. If it isn't, putIfAbsent will
    // return null, and we should return the current instance.
    FullyQualifiedName cached = instanceCache.putIfAbsent(name, name);
    if (cached == null) {
      return name;
    } else {
      cacheHit.incrementAndGet();
      return cached;
    }
  }

  public static FullyQualifiedName fromProto(SerializeFormat.DataKey protoKey) {
    return of(
        protoKey.getKeyPackage(),
        protoKey.getQualifiersList(),
        ResourceType.valueOf(protoKey.getResourceType()),
        protoKey.getKeyValue());
  }

  public static void logCacheUsage(Logger logger) {
    logger.fine(
        String.format(
            "Total FullyQualifiedName instance cache hits %s out of %s",
            cacheHit.intValue(),
            instanceCache.size()));
  }

  private final String pkg;
  private final ImmutableList<String> qualifiers;
  private final ResourceType resourceType;
  private final String resourceName;

  /**
   * Returns a string path representation of the FullyQualifiedName.
   *
   * Non-values Android Resource have a well defined file layout: From the resource directory, they
   * reside in &lt;resource type&gt;[-&lt;qualifier&gt;]/&lt;resource name&gt;[.extension]
   *
   * @param source The original source of the file-based resource's FullyQualifiedName
   * @return A string representation of the FullyQualifiedName with the provided extension.
   */
  public String toPathString(Path source) {
    String sourceExtension = FullyQualifiedName.Factory.getSourceExtension(source);
    return Paths.get(
            DASH_JOINER.join(
                ImmutableList.<String>builder()
                    .add(resourceType.getName())
                    .addAll(qualifiers)
                    .build()),
            resourceName + sourceExtension)
        .toString();
  }

  @Override
  public String toPrettyString() {
    // TODO(corysmith): Add package when we start tracking it.
    return String.format(
        "%s/%s",
        DASH_JOINER.join(
            ImmutableList.<String>builder().add(resourceType.getName()).addAll(qualifiers).build()),
        resourceName);
  }

  /**
   * Returns the string path representation of the values directory and qualifiers.
   *
   * Certain resource types live in the "values" directory. This will calculate the directory and
   * ensure the qualifiers are represented.
   */
  // TODO(corysmith): Combine this with toPathString to clean up the interface of FullyQualifiedName
  // logically, the FullyQualifiedName should just be able to provide the relative path string for
  // the resource.
  public String valuesPath() {
    return Paths.get(
            DASH_JOINER.join(
                ImmutableList.<String>builder().add("values").addAll(qualifiers).build()),
            "values.xml")
        .toString();
  }

  public String name() {
    return resourceName;
  }

  public ResourceType type() {
    return resourceType;
  }

  private FullyQualifiedName(
      String pkg,
      ImmutableList<String> qualifiers,
      ResourceType resourceType,
      String resourceName) {
    this.pkg = pkg;
    this.qualifiers = qualifiers;
    this.resourceType = resourceType;
    this.resourceName = resourceName;
  }

  /** Creates a FullyQualifiedName from this one with a different package. */
  @CheckReturnValue
  public FullyQualifiedName replacePackage(String newPackage) {
    if (pkg.equals(newPackage)) {
      return this;
    }
    return of(newPackage, qualifiers, resourceType, resourceName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(pkg, qualifiers, resourceType, resourceName);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof FullyQualifiedName)) {
      return false;
    }
    FullyQualifiedName other = getClass().cast(obj);
    return Objects.equals(pkg, other.pkg)
        && Objects.equals(resourceType, other.resourceType)
        && Objects.equals(resourceName, other.resourceName)
        && Objects.equals(qualifiers, other.qualifiers);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("pkg", pkg)
        .add("qualifiers", qualifiers)
        .add("resourceType", resourceType)
        .add("resourceName", resourceName)
        .toString();
  }

  @Override
  public int compareTo(FullyQualifiedName other) {
    if (!pkg.equals(other.pkg)) {
      return pkg.compareTo(other.pkg);
    }
    if (!resourceType.equals(other.resourceType)) {
      return resourceType.compareTo(other.resourceType);
    }
    if (!resourceName.equals(other.resourceName)) {
      return resourceName.compareTo(other.resourceName);
    }
    if (!qualifiers.equals(other.qualifiers)) {
      if (qualifiers.size() != other.qualifiers.size()) {
        return qualifiers.size() - other.qualifiers.size();
      }
      // This works because the qualifiers are always in an ordered sequence.
      return qualifiers.toString().compareTo(other.qualifiers.toString());
    }
    return 0;
  }

  @Override
  public void serializeTo(OutputStream out, int valueSize) throws IOException {
    toSerializedBuilder().setValueSize(valueSize).build().writeDelimitedTo(out);
  }

  public SerializeFormat.DataKey.Builder toSerializedBuilder() {
    return SerializeFormat.DataKey.newBuilder()
        .setKeyPackage(pkg)
        .setResourceType(resourceType.getName().toUpperCase())
        .addAllQualifiers(qualifiers)
        .setKeyValue(resourceName);
  }
}
