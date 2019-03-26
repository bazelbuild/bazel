package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.remote.blobstore.CombinedDiskHttpBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Options;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;


@RunWith(JUnit4.class)
public class SimpleBlobStoreFactoryTest {

  @Rule
  public ExpectedException expectedEx = ExpectedException.none();

  @Mock
  private Path workingDirectory;

  @Mock
  private Path relativeDirectory;

  @Before
  public final void setUp() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void createCombinedCache() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(options.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(true);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds */ null, workingDirectory);

    assertThat(blobStore).isNotNull();
    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
  }

  @Test
  public void createCombinedCache_whenWorkingDirectoryDoesNotExist() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(options.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(false);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds */ null, workingDirectory);

    assertThat(blobStore).isNotNull();
    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
    verify(relativeDirectory, times(1)).createDirectoryAndParents();
  }

  @Test
  public void createCombinedCache_whenWorkingDirectoryNull_throwsNPE() throws IOException {
    // interesting case: workingDirectory = null -> NPE.
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    expectedEx.expect(NullPointerException.class);

    SimpleBlobStoreFactory.create(options, /* creds */ null, /* workingDirectory */ null);
  }

  @Test
  public void createHttpCacheWithProxy() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.remoteCacheProxy = "unix://some-proxy";
    options.diskCache = null;

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds */ null, workingDirectory);

    assertThat(blobStore).isNotNull();
    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createHttpCacheWithProxy_notSupportedProtocol_throwsException() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.remoteCacheProxy = "bad-proxy";
    options.diskCache = null;

    expectedEx.expect(Exception.class);
    expectedEx.expectMessage("Remote cache proxy unsupported: bad-proxy");

    SimpleBlobStoreFactory.create(options, /* creds */ null, workingDirectory);
  }

  @Test
  public void createHttpCacheWithoutProxy() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = null;

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds */ null, workingDirectory);

    assertThat(blobStore).isNotNull();
    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createDiskCache() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(options.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(true);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds */ null, workingDirectory);

    assertThat(blobStore).isNotNull();
    assertThat(blobStore).isInstanceOf(OnDiskBlobStore.class);
  }

  @Test
  public void isRemoteCacheOptions_true_whenHttpCache() {
    RemoteOptions options = new RemoteOptions();
    options.remoteHttpCache = "http://doesnotexist:90";
    options.diskCache = null;

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_true_whenDiskCache() {
    RemoteOptions options = new RemoteOptions();
    options.remoteHttpCache = null;
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_true_whenHttpAndDiskCache() {
    RemoteOptions options = new RemoteOptions();
    options.remoteHttpCache = "http://doesnotexist:90";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_false_whenDiskCacheOptionEmpty() {
    RemoteOptions options = new RemoteOptions();
    options.diskCache = PathFragment.EMPTY_FRAGMENT;
    options.remoteHttpCache = null;

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_false_whenRemoteHttpCacheEmpty() {
    RemoteOptions options = new RemoteOptions();
    options.remoteHttpCache = "";

    // todo #7650: flip to false
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_false_whenDefaultOptions() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isFalse();
  }
}