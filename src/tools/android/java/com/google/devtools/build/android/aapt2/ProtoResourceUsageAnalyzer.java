// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.aapt2;

import static com.android.SdkConstants.ATTR_DISCARD;
import static com.android.SdkConstants.ATTR_KEEP;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;

import com.android.aapt.Resources;
import com.android.build.gradle.tasks.ResourceUsageAnalyzer;
import com.android.resources.ResourceFolderType;
import com.android.resources.ResourceType;
import com.android.tools.lint.checks.ResourceUsageModel;
import com.android.tools.lint.checks.ResourceUsageModel.Resource;
import com.android.tools.lint.detector.api.LintUtils;
import com.android.utils.XmlUtils;
import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.android.aapt2.ProtoApk.ManifestVisitor;
import com.google.devtools.build.android.aapt2.ProtoApk.ReferenceVisitor;
import com.google.devtools.build.android.aapt2.ProtoApk.ResourcePackageVisitor;
import com.google.devtools.build.android.aapt2.ProtoApk.ResourceValueVisitor;
import com.google.devtools.build.android.aapt2.ProtoApk.ResourceVisitor;
import com.sun.org.apache.xerces.internal.dom.AttrImpl;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Attr;
import org.w3c.dom.DOMException;

/** A resource usage analyzer tha functions on apks in protocol buffer format. */
public class ProtoResourceUsageAnalyzer extends ResourceUsageAnalyzer {

  private static final Logger logger = Logger.getLogger(ProtoResourceUsageAnalyzer.class.getName());
  private final Set<String> resourcePackages;
  private final Path rTxt;
  private final Path mapping;
  private final Path resourcesConfigFile;

  public ProtoResourceUsageAnalyzer(
      Set<String> resourcePackages,
      Path rTxt,
      Path mapping,
      Path resourcesConfigFile,
      Path logFile)
      throws DOMException, ParserConfigurationException {
    super(resourcePackages, null, null, null, null, null, logFile);
    this.resourcePackages = resourcePackages;
    this.rTxt = rTxt;
    this.mapping = mapping;
    this.resourcesConfigFile = resourcesConfigFile;
  }

  private static Resource parse(ResourceUsageModel model, String resourceTypeAndName) {
    final Iterator<String> iterator = Splitter.on('/').split(resourceTypeAndName).iterator();
    Preconditions.checkArgument(
        iterator.hasNext(), "%s invalid resource name", resourceTypeAndName);
    ResourceType resourceType = ResourceType.getEnum(iterator.next());
    Preconditions.checkArgument(
        iterator.hasNext(), "%s invalid resource name", resourceTypeAndName);
    return model.getResource(resourceType, iterator.next());
  }

  /**
   * Calculate and removes unused resource from the {@link ProtoApk}.
   *
   * @param apk An apk in the aapt2 proto format.
   * @param classes The associated classes for the apk.
   * @param destination Where to write the reduced resources.
   * @param toolAttributes A map of the tool attributes designating resources to keep or discard.
   */
  @CheckReturnValue
  public ProtoApk shrink(
      ProtoApk apk, Path classes, Path destination, ListMultimap<String, String> toolAttributes)
      throws IOException {

    // Set the usage analyzer as parent to make sure that the usage log contains the subclass data.
    logger.setParent(Logger.getLogger(ResourceUsageAnalyzer.class.getName()));
    // record resources and manifest
    apk.visitResources(
        // First, collect all declarations using the declaration visitor.
        // This allows the model to start with a defined set of resources to build the reference
        // graph on.
        apk.visitResources(new ResourceDeclarationVisitor(model())).toUsageVisitor());

    try {
      // TODO(b/112810967): Remove reflection hack.
      final Method parseResourceTxtFile =
          ResourceUsageAnalyzer.class.getDeclaredMethod(
              "parseResourceTxtFile", Path.class, Set.class);
      parseResourceTxtFile.setAccessible(true);
      parseResourceTxtFile.invoke(this, rTxt, resourcePackages);
      final Method recordMapping =
          ResourceUsageAnalyzer.class.getDeclaredMethod("recordMapping", Path.class);
      recordMapping.setAccessible(true);
      recordMapping.invoke(this, mapping);
    } catch (ReflectiveOperationException e) {
      throw new RuntimeException(e);
    }
    recordClassUsages(classes);

    toolAttributes.entries().stream()
        .filter(entry -> entry.getKey().equals(ATTR_KEEP) || entry.getKey().equals(ATTR_DISCARD))
        .map(entry -> createSimpleAttr(entry.getKey(), entry.getValue()))
        .forEach(attr -> model().recordToolsAttributes(attr));
    model().processToolsAttributes();

    keepPossiblyReferencedResources();

    final List<Resource> resources = model().getResources();
    final ImmutableListMultimap<ResourceTypeAndJavaName, String> unJavafiedNames =
        getUnJavafiedResourceNames(apk);

    ImmutableList<String> resourceConfigs =
        resources.stream()
            .filter(Resource::isKeep)
            .flatMap(
                r ->
                    // aapt2 expects the original resource names, not the Java-sanitized names.
                    //
                    // "Resource" is written in such a way so that Resource#getField (Java) and
                    // Resource#getUrl (actual name) cannot both work, so we have to undo that.
                    unJavafiedNames
                        .get(ResourceTypeAndJavaName.of(r.type.getName(), r.name))
                        .stream()
                        .map(orig -> new Resource(r.type, orig, r.value)))
            .map(r -> String.format("%s/%s#no_collapse", r.type.getName(), r.name))
            .collect(toImmutableList());
    Files.write(resourcesConfigFile, resourceConfigs, StandardCharsets.UTF_8);

    List<Resource> roots =
        resources.stream().filter(r -> r.isKeep() || r.isReachable()).collect(toList());

    final Set<Resource> reachable = findReachableResources(roots);
    return apk.copy(
        destination,
        (resourceType, name) -> reachable.contains(model().getResource(resourceType, name)));
  }

  private Set<Resource> findReachableResources(List<Resource> roots) {
    final Multimap<Resource, Resource> referenceLog = LinkedHashMultimap.create();
    Deque<Resource> queue = new ArrayDeque<>(roots);
    final Set<Resource> reachable = new LinkedHashSet<>();
    while (!queue.isEmpty()) {
      Resource resource = queue.pop();
      if (resource.references != null) {
        resource.references.forEach(
            r -> {
              referenceLog.put(r, resource);
              // add if it has not been marked reachable, therefore processed.
              if (!reachable.contains(r)) {
                queue.add(r);
              }
            });
      }
      // if we see it, it is reachable.
      reachable.add(resource);
    }

    // dump resource reference map:
    final StringBuilder keptResourceLog = new StringBuilder();
    referenceLog
        .asMap()
        .forEach(
            (resource, referencesTo) ->
                keptResourceLog
                    .append(printResource(resource))
                    .append(" => [")
                    .append(
                        referencesTo.stream()
                            .map(ProtoResourceUsageAnalyzer::printResource)
                            .collect(joining(", ")))
                    .append("]\n"));

    logger.fine("Kept resource references:\n" + keptResourceLog);

    return reachable;
  }

  private static String printResource(Resource res) {
    return String.format(
        "{%s[isRoot: %s] = %s}",
        res.getUrl(), res.isReachable() || res.isKeep(), "0x" + Integer.toHexString(res.value));
  }

  private static final class ResourceDeclarationVisitor implements ResourceVisitor {

    private final ResourceShrinkerUsageModel model;
    private final Set<Integer> packageIds = new LinkedHashSet<>();

    private ResourceDeclarationVisitor(ResourceShrinkerUsageModel model) {
      this.model = model;
    }

    @Nullable
    @Override
    public ManifestVisitor enteringManifest() {
      return null;
    }

    @Override
    public ResourcePackageVisitor enteringPackage(int pkgId, String packageName) {
      packageIds.add(pkgId);
      return (typeId, resourceType) ->
          (name, resourceId) -> {
            String hexId =
                String.format(
                    "0x%s", Integer.toHexString(((pkgId << 24) | (typeId << 16) | resourceId)));
            model.addDeclaredResource(resourceType, LintUtils.getFieldName(name), hexId, true);
            // Skip visiting the definition when collecting declarations.
            return null;
          };
    }

    ResourceUsageVisitor toUsageVisitor() {
      return new ResourceUsageVisitor(model, ImmutableSet.copyOf(packageIds));
    }
  }

  private static final class ResourceUsageVisitor implements ResourceVisitor {

    private final ResourceShrinkerUsageModel model;
    private final ImmutableSet<Integer> packageIds;

    private ResourceUsageVisitor(
        ResourceShrinkerUsageModel model, ImmutableSet<Integer> packageIds) {
      this.model = model;
      this.packageIds = packageIds;
    }

    @Override
    public ManifestVisitor enteringManifest() {
      return new ManifestVisitor() {
        @Override
        public void accept(String name) {
          ResourceUsageModel.markReachable(model.getResourceFromUrl(name));
        }

        @Override
        public void accept(int value) {
          ResourceUsageModel.markReachable(model.getResource(value));
        }
      };
    }

    @Override
    public ResourcePackageVisitor enteringPackage(int pkgId, String packageName) {
      return (typeId, resourceType) ->
          (name, resourceId) ->
              new ResourceUsageValueVisitor(
                  model, model.getResource(resourceType, name), packageIds);
    }
  }

  private static final class ResourceUsageValueVisitor implements ResourceValueVisitor {

    private final ResourceUsageModel model;
    private final Resource declaredResource;
    private final ImmutableSet<Integer> packageIds;

    private ResourceUsageValueVisitor(
        ResourceUsageModel model, Resource declaredResource, ImmutableSet<Integer> packageIds) {
      this.model = model;
      this.declaredResource = declaredResource;
      this.packageIds = packageIds;
    }

    @Override
    public ReferenceVisitor entering(Path path) {
      return this;
    }

    @Override
    public void acceptOpaqueFileType(Path path) {
      try {
        String pathString = path.toString();
        if (pathString.endsWith(".js")) {
          model.tokenizeJs(
              declaredResource, new String(Files.readAllBytes(path), StandardCharsets.UTF_8));
        } else if (pathString.endsWith(".css")) {
          model.tokenizeCss(
              declaredResource, new String(Files.readAllBytes(path), StandardCharsets.UTF_8));
        } else if (pathString.endsWith(".html")) {
          model.tokenizeHtml(
              declaredResource, new String(Files.readAllBytes(path), StandardCharsets.UTF_8));
        } else if (pathString.endsWith(".xml")) {
          // Force parsing of raw xml files to get any missing keep attributes.
          // The tool keep and discard attributes are held in raw files.
          // There is already processing to handle this, but there has been flakiness.
          // This step is to ensure as much stability as possible until the flakiness can be
          // diagnosed.
          model.recordResourceReferences(
              ResourceFolderType.getTypeByName(declaredResource.type.getName()),
              XmlUtils.parseDocumentSilently(
                  new String(Files.readAllBytes(path), StandardCharsets.UTF_8), true),
              declaredResource);

        } else {
          // Path is a reference to the apk zip -- unpack it before getting a file reference.
          model.tokenizeUnknownBinary(
              declaredResource,
              Files.copy(
                      path,
                      Files.createTempFile("binary-resource", null),
                      StandardCopyOption.REPLACE_EXISTING)
                  .toFile());
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public void accept(String name) {
      declaredResource.addReference(parse(model, name));
    }

    @Override
    public void accept(int value) {
      if (isInDeclaredPackages(value)) { // ignore references outside of scanned packages.
        declaredResource.addReference(model.getResource(value));
      }
    }

    /** Tests if the id is in any of the scanned packages. */
    private boolean isInDeclaredPackages(int value) {
      return packageIds.contains(value >> 24);
    }
  }

  @VisibleForTesting
  public static Attr createSimpleAttr(String simpleName, String simpleValue) {
    return new AttrImpl() {
      @Override
      public String getLocalName() {
        return simpleName;
      }

      @Override
      public String getValue() {
        return simpleValue;
      }
    };
  }

  /**
   * Maps resource type and Java-fied name (i.e. dots converted to underscores) to the original
   * name(s).
   */
  // This is used to work around the fact that (a) ResourceUsageModel throws away the original
  // names, and (b) ResourceUsageModel is from an external library not synced with Bazel.
  //
  // Using a multimap because LintUtils.getFieldName is a many-to-one mapping, meaning that multiple
  // resources could have the same Java name.  Assuming that the rest of the build system doesn't
  // blow up, neither should we.
  static ImmutableListMultimap<ResourceTypeAndJavaName, String> getUnJavafiedResourceNames(
      ProtoApk apk) throws IOException {
    ImmutableListMultimap.Builder<ResourceTypeAndJavaName, String> unJavafiedNames =
        ImmutableListMultimap.builder();
    for (Resources.Package pkg : apk.getResourceTable().getPackageList()) {
      for (Resources.Type type : pkg.getTypeList()) {
        for (Resources.Entry entry : type.getEntryList()) {
          String originalName = entry.getName();
          String javafiedName = LintUtils.getFieldName(originalName);
          unJavafiedNames.put(
              ResourceTypeAndJavaName.of(type.getName(), javafiedName), originalName);
        }
      }
    }
    return unJavafiedNames.build();
  }

  @AutoValue
  abstract static class ResourceTypeAndJavaName {
    abstract String type();

    abstract String javaName();

    static ResourceTypeAndJavaName of(String type, String javaName) {
      return new AutoValue_ProtoResourceUsageAnalyzer_ResourceTypeAndJavaName(type, javaName);
    }
  }
}
