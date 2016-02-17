// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import org.apache.maven.model.Repository;
import org.apache.maven.repository.internal.MavenRepositorySystemUtils;
import org.eclipse.aether.AbstractRepositoryListener;
import org.eclipse.aether.DefaultRepositorySystemSession;
import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.RepositorySystemSession;
import org.eclipse.aether.connector.basic.BasicRepositoryConnectorFactory;
import org.eclipse.aether.impl.DefaultServiceLocator;
import org.eclipse.aether.repository.LocalRepository;
import org.eclipse.aether.repository.RemoteRepository;
import org.eclipse.aether.spi.connector.RepositoryConnectorFactory;
import org.eclipse.aether.spi.connector.transport.TransporterFactory;
import org.eclipse.aether.transfer.AbstractTransferListener;
import org.eclipse.aether.transport.file.FileTransporterFactory;
import org.eclipse.aether.transport.http.HttpTransporterFactory;

/**
 * Connections to Maven repositories.
 */
public class MavenConnector {
  public static final String MAVEN_CENTRAL_URL = "https://repo1.maven.org/maven2/";

  private final String localRepositoryPath;

  public MavenConnector(String localRepositoryPath) {
    this.localRepositoryPath = localRepositoryPath;
  }

  public RepositorySystemSession newRepositorySystemSession(RepositorySystem system) {
    DefaultRepositorySystemSession session = MavenRepositorySystemUtils.newSession();
    LocalRepository localRepo = new LocalRepository(localRepositoryPath);
    session.setLocalRepositoryManager(system.newLocalRepositoryManager(session, localRepo));
    session.setTransferListener(new AbstractTransferListener() {});
    session.setRepositoryListener(new AbstractRepositoryListener() {});
    return session;
  }

  public RepositorySystem newRepositorySystem() {
    DefaultServiceLocator locator = MavenRepositorySystemUtils.newServiceLocator();
    locator.addService(RepositoryConnectorFactory.class, BasicRepositoryConnectorFactory.class);
    locator.addService(TransporterFactory.class, FileTransporterFactory.class);
    locator.addService(TransporterFactory.class, HttpTransporterFactory.class);
    return locator.getService(RepositorySystem.class);
  }

  /**
   * How is this not a built-in for aether?
   */
  public static RemoteRepository getMavenCentralRemote() {
    return new RemoteRepository.Builder(
        "central", "default", MAVEN_CENTRAL_URL).build();
  }

  public static Repository getMavenCentral() {
    Repository repository = new Repository();
    repository.setId("central");
    repository.setName("default");
    repository.setUrl(MAVEN_CENTRAL_URL);
    return repository;
  }

}
