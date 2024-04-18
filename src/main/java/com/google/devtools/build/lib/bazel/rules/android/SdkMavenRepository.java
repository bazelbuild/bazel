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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * A collection of .pom contents that reference versioned archive files with dependencies.
 */
final class SdkMavenRepository {
  private enum PackagingType {
    AAR {
      @Override
      ImmutableList<String> createRule(
          String name, String archiveLabel, ImmutableList<String> dependencyLabels) {
        ImmutableList.Builder<String> ruleLines = new ImmutableList.Builder<>();
        ruleLines.add("aar_import(");
        ruleLines.add("    name = '" + name + "',");
        ruleLines.add("    aar = '" + archiveLabel + "',");
        ruleLines.add("    exports = [");
        for (String dependencyLabel : dependencyLabels) {
          ruleLines.add("        '" + dependencyLabel + "',");
        }
        ruleLines.add("    ],");
        ruleLines.add(")");
        ruleLines.add("");
        return ruleLines.build();
      }
    },
    JAR {
      @Override
      ImmutableList<String> createRule(
          String name, String archiveLabel, ImmutableList<String> dependencyLabels) {
        ImmutableList.Builder<String> ruleLines = new ImmutableList.Builder<>();
        ruleLines.add("java_import(");
        ruleLines.add("    name = '" + name + "',");
        ruleLines.add("    jars = ['" + archiveLabel + "'],");
        ruleLines.add("    exports = [");
        for (String dependencyLabel : dependencyLabels) {
          ruleLines.add("        '" + dependencyLabel + "',");
        }
        ruleLines.add("    ],");
        ruleLines.add(")");
        ruleLines.add("");
        return ruleLines.build();
      }
    },
    UNKNOWN {
      @Override
      ImmutableList<String> createRule(String name, String archiveLabel,
          ImmutableList<String> dependencyLabels) {
        ImmutableList.Builder<String> ruleLines = new ImmutableList.Builder<>();
        ruleLines.add("genrule(");
        ruleLines.add("    name = '" + name + "',");
        ruleLines.add("    outs = ['ignored_" + name + "'],");
        ruleLines.add("    cmd = 'echo Bazel does not recognize the Maven packaging type for: \""
            + archiveLabel + "\"; exit 1',");
        ruleLines.add(")");
        ruleLines.add("");
        return ruleLines.build();
      }
    };

    static PackagingType getPackagingType(String name) {
      for (PackagingType packagingType : PackagingType.values()) {
        if (packagingType.name().equalsIgnoreCase(name)) {
          return packagingType;
        }
      }
      return UNKNOWN;
    }

    abstract ImmutableList<String> createRule(
        String name, String archiveLabel, ImmutableList<String> dependencyLabels);
  }

  private final ImmutableSortedSet<Pom> poms;
  private final ImmutableSet<MavenCoordinate> allKnownCoordinates;

  private SdkMavenRepository(ImmutableSortedSet<Pom> poms) {
    this.poms = poms;
    ImmutableSet.Builder<MavenCoordinate> coordinates = new ImmutableSet.Builder<>();
    for (Pom pom : poms) {
      if (!PackagingType.getPackagingType(pom.packaging()).equals(PackagingType.UNKNOWN)) {
        coordinates.add(pom.mavenCoordinate());
      }
    }
    allKnownCoordinates = coordinates.build();
  }

  /**
   * Parses a set of maven repository directory trees looking for and parsing .pom files.
   */
  static SdkMavenRepository create(Iterable<Path> mavenRepositories) throws IOException {
    Collection<Path> pomPaths = new ArrayList<>();
    for (Path mavenRepository : mavenRepositories) {
      pomPaths.addAll(
          FileSystemUtils.traverseTree(mavenRepository, path -> path.toString().endsWith(".pom")));
    }

    ImmutableSortedSet.Builder<Pom> poms =
        new ImmutableSortedSet.Builder<>(Ordering.usingToString());
    for (Path pomPath : pomPaths) {
      try {
        Pom pom = Pom.parse(pomPath);
        if (pom != null) {
          poms.add(pom);
        }
      } catch (ParserConfigurationException | SAXException e) {
        throw new IOException(e);
      }
    }
    return new SdkMavenRepository(poms.build());
  }

  /**
   * Creates BUILD files at {@code @<android sdk>/<group id>/BUILD} containing aar_import and
   * java_import rules with dependencies as {@code exports}. The targets are named
   * {@code @<android sdk>//<group id>:<artifact id>-<version id>}.
   */
  void writeBuildFiles(Path outputDirectory) throws IOException {
    for (Pom pom : poms) {
      Path buildFilePath = outputDirectory.getRelative(pom.mavenCoordinate().groupId() + "/BUILD");

      if (!buildFilePath.getParentDirectory().exists()) {
        buildFilePath.getParentDirectory().createDirectory();
      }

      if (!buildFilePath.exists()) {
        FileSystemUtils.writeContentAsLatin1(
            buildFilePath, "package(default_visibility = [\"//visibility:public\"])\n\n");
      }

      ImmutableList.Builder<String> dependencyLabels = new ImmutableList.Builder<>();
      for (MavenCoordinate dependencyCoordinate : pom.dependencyCoordinates()) {
        // Filter out dependencies that are not present in the Maven repository or have unknown
        // packaging types.
        if (allKnownCoordinates.contains(dependencyCoordinate)) {
          dependencyLabels.add(dependencyCoordinate.targetLabel());
        }
      }

      ImmutableList<String> ruleLines = PackagingType.getPackagingType(pom.packaging())
          .createRule(
              pom.mavenCoordinate().targetName(),
              pom.archiveLabel(outputDirectory),
              dependencyLabels.build());
      FileSystemUtils.appendLinesAs(buildFilePath, StandardCharsets.ISO_8859_1, ruleLines);
    }
  }

  /**
   * Creates the contents of the exports_files rule list containing all of the archives specified by
   * the pom files in the Maven repositories.
   */
  String getExportsFiles(Path outputDirectory) {
    StringBuilder exportedFiles = new StringBuilder();
    for (Pom pom : poms) {
      exportedFiles.append(String.format(
          "    '%s',\n", pom.archivePath().relativeTo(outputDirectory).getPathString()));
    }
    return exportedFiles.toString();
  }

  /**
   * The relevant contents of a .pom file needed to populate BUILD files for aars and jars in the
   * Android SDK extras maven repositories.
   */
  @AutoValue
  abstract static class Pom {
    private static final String DEFAULT_PACKAGING = "jar";

    static Pom parse(Path path) throws IOException, ParserConfigurationException, SAXException {
      Document pomDocument = null;
      try (InputStream in = path.getInputStream()) {
        pomDocument = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);
      }
      Node packagingNode = pomDocument.getElementsByTagName("packaging").item(0);
      String packaging = packagingNode == null ? DEFAULT_PACKAGING : packagingNode.getTextContent();
      MavenCoordinate coordinate = MavenCoordinate.create(
          pomDocument.getElementsByTagName("groupId").item(0).getTextContent(),
          pomDocument.getElementsByTagName("artifactId").item(0).getTextContent(),
          pomDocument.getElementsByTagName("version").item(0).getTextContent());

      ImmutableSortedSet.Builder<MavenCoordinate> dependencyCoordinates =
          new ImmutableSortedSet.Builder<>(Ordering.usingToString());
      NodeList dependencies = pomDocument.getElementsByTagName("dependency");
      for (int i = 0; i < dependencies.getLength(); i++) {
        if (dependencies.item(i) instanceof Element dependency) {
          dependencyCoordinates.add(MavenCoordinate.create(
              dependency.getElementsByTagName("groupId").item(0).getTextContent(),
              dependency.getElementsByTagName("artifactId").item(0).getTextContent(),
              dependency.getElementsByTagName("version").item(0).getTextContent()));
        }
      }

      return new AutoValue_SdkMavenRepository_Pom(
          path, packaging, coordinate, dependencyCoordinates.build());
    }

    abstract Path path();

    abstract String packaging();

    abstract MavenCoordinate mavenCoordinate();

    abstract ImmutableSortedSet<MavenCoordinate> dependencyCoordinates();

    String name() {
      String pomFilename = path().getBaseName();
      return pomFilename.substring(0, pomFilename.lastIndexOf(".pom"));
    }

    Path archivePath() {
      return path().getParentDirectory().getRelative(name() + "." + packaging());
    }

    /** The label for the .aar or .jar file in the repository. */
    String archiveLabel(Path outputDirectory) {
      return "//:" + archivePath().relativeTo(outputDirectory);
    }
  }

  /**
   * A 3-tuple of group id, artifact id and version used to identify Maven targets.
   */
  @AutoValue
  abstract static class MavenCoordinate {
    static MavenCoordinate create(String groupId, String artifactId, String version) {
      return new AutoValue_SdkMavenRepository_MavenCoordinate(groupId, artifactId, version);
    }

    abstract String groupId();

    abstract String artifactId();

    abstract String version();

    /** The target name for the java_import or aar_import for the Maven coordinate. */
    String targetName() {
      return artifactId() + "-" + version();
    }

    /** The target label for the java_import or aar_import for the Maven coordinate. */
    String targetLabel() {
      return "//" + groupId() + ":" + targetName();
    }
  }
}
