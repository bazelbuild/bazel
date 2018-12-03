package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.devtools.build.lib.bazel.repository.downloader.CredentialsProvider.Credentials;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.UUID;

@RunWith(JUnit4.class)
public class NetrcCredentialsProviderTest {

    @Test(expected = NetrcCredentialsProviderException.class)
    public void getCredentials_ThrowsFileNotFoundExceptionOnNonExistingGivenNetrcFilePath() {
        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(Paths.get("non_existing"));

        credentialsProvider.getCredentials(HOST);
    }

    @Test
    public void getCredentials_ReturnsEmptyResultWhenNetrcFileIsEmpty() {
        givenNetrcFileExistsButEmpty();

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.empty(), actual);
    }

    @Test
    public void getCredentials_ReturnsCredentialsWithEmptyUserAndPasswordWhenMachineFoundButWithoutLoginAndPassword() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRowWihMachine(HOST));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials()), actual);
    }

    @Test
    public void getCredentials_ReturnsCredentialsWithUserAndEmptyPasswordWhenMachineFoundWithLoginButWithoutPassword() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRowWithMachineAndLogin(HOST, LOGIN));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, null)), actual);
    }

    @Test
    public void getCredentials_ReturnsCredentialsWithPasswordAndEmptyUserWhenMachineFoundWithPasswordButWithoutLogin() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRowWithMachineAndPassword(HOST, PASS));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(null, PASS)), actual);
    }

    @Test
    public void getCredentials_ReturnsCredentialsWithUserAndPassword() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST, LOGIN, PASS));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);
    }

    @Test
    public void getCredentials_ReturnsCredentialsForSubDomain() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST, LOGIN, PASS));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(SUBDOMAIN);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);
    }

    @Test
    public void getCredentials_ReturnsCredentialsForFirstMatchOfMachine() {
        givenNetrcFileWith(NETRC_FILE_PATH,
                toNetrcRow(HOST, LOGIN, PASS),
                toNetrcRow(SUBDOMAIN, LOGIN2, PASS2)
        );

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(SUBDOMAIN);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);
    }

    @Test
    public void getCredentials_FindsHostInTheMiddle() {
        givenNetrcFileWith(NETRC_FILE_PATH,
                toNetrcRowWithNewLinesAndTabs(HOST, LOGIN, PASS),
                toNetrcRow(HOST2, LOGIN2, PASS2),
                toNetrcRow(HOST3, LOGIN3, PASS3)
        );

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST2);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN2, PASS2)), actual);
    }

    @Test
    public void getCredentials_SupportsDefinitionsWithNewLines() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRowWithNewLinesAndTabs(HOST, LOGIN, PASS));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);
    }

    @Test
    public void getCredentials_ReturnEmptyResultWhenHostIsNotFound() {
        givenNetrcFileWith(NETRC_FILE_PATH,
                toNetrcRow(HOST, LOGIN, PASS),
                toNetrcRow(HOST2, LOGIN2, PASS2)
        );

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST3);
        Assert.assertEquals(Optional.empty(), actual);
    }

    @Test
    public void getCredentials_CachingNetrcFileContentAfterFirstRead() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST, LOGIN, PASS));

        CredentialsProvider credentialsProvider = new NetrcCredentialsProvider(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);

        deleteFile(NETRC_FILE_PATH);
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST2, LOGIN2, PASS2));

        Optional<Credentials> actualCached = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actualCached);
    }

    @Test
    public void getInstance_ReturnsTheSameObjectForSamePath() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST, LOGIN, PASS));

        CredentialsProvider credentialsProvider = NetrcCredentialsProvider.getInstance(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);

        // now the contents of netrc are cached within the object
        // we will create a new netrc file, and test that it is not read again, when getting the instance again
        // via getInstance with the same path
        deleteFile(NETRC_FILE_PATH);
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST2, LOGIN2, PASS2));

        CredentialsProvider credentialsProviderAgain = NetrcCredentialsProvider.getInstance(NETRC_FILE_PATH);
        Optional<Credentials> actualCached = credentialsProviderAgain.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actualCached);
    }

    @Test
    public void getInstance_ReturnsDifferentObjectsForDifferentPaths() {
        givenNetrcFileWith(NETRC_FILE_PATH, toNetrcRow(HOST, LOGIN, PASS));
        givenNetrcFileWith(NETRC_FILE_PATH2, toNetrcRow(HOST2, LOGIN2, PASS2));

        CredentialsProvider credentialsProvider = NetrcCredentialsProvider.getInstance(NETRC_FILE_PATH);

        Optional<Credentials> actual = credentialsProvider.getCredentials(HOST);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN, PASS)), actual);

        CredentialsProvider credentialsProviderAgain = NetrcCredentialsProvider.getInstance(NETRC_FILE_PATH2);
        Optional<Credentials> actualCached = credentialsProviderAgain.getCredentials(HOST2);
        Assert.assertEquals(Optional.of(new Credentials(LOGIN2, PASS2)), actualCached);
    }

    @After
    public void afterEach() {
        deleteFile(NETRC_FILE_PATH);
        deleteFile(NETRC_FILE_PATH2);
    }

    private void deleteFile(Path path) {
        try {
            Files.delete(path);
        } catch (NoSuchFileException e) {
            // do nothing
        } catch (IOException e) {
            Assert.fail(String.format("failed to delete file %s", NETRC_FILE_PATH));
            e.printStackTrace();
        }
    }

    private void givenNetrcFileExistsButEmpty() {
        try {
            Files.createFile(NETRC_FILE_PATH);
        } catch (IOException e) {
            Assert.fail("could not create .netrc file");
        }
    }

    private void givenNetrcFileWith(Path netrcFilePath, String... netrcRows) {
        StringBuilder fileContent = new StringBuilder();
        for (String row : netrcRows) {
            fileContent.append(row).append(System.lineSeparator());
        }
        try {
            Files.write(netrcFilePath, fileContent.toString().getBytes(Charset.forName("UTF-8")));
        } catch (IOException e) {
            Assert.fail("could not create .netrc file");
        }
    }

    private String toNetrcRow(String host, String login, String password) {
        String s = String.format("machine %s login %s password %s", host, login, password);
        System.out.println(s);
        return s;
    }

    private String toNetrcRowWithNewLinesAndTabs(String host, String login, String password) {
        String s = String.format("machine %s%s\tlogin %s%s\t\tpassword %s",
                host, System.lineSeparator(), login, System.lineSeparator(), password);
        System.out.println(s);
        return s;
    }


    private String toNetrcRowWihMachine(String machine) {
        return String.format("machine %s", machine);
    }

    private String toNetrcRowWithMachineAndLogin(String host, String login) {
        return String.format("machine %s login %s", host, login);
    }

    private String toNetrcRowWithMachineAndPassword(String host, String password) {
        return String.format("machine %s password %s", host, password);
    }

    private static final Path NETRC_FILE_PATH = Paths.get("./.netrc");
    private static final Path NETRC_FILE_PATH2 = Paths.get("./.netrc2");
    private static final String HOST = "some.host" + UUID.randomUUID().toString();
    private static final String HOST2 = "some.host2" + UUID.randomUUID().toString();
    private static final String HOST3 = "some.host3" + UUID.randomUUID().toString();
    private static final String SUBDOMAIN = "sub." + HOST;
    private static final String LOGIN = "login_" + UUID.randomUUID().toString();
    private static final String LOGIN2 = "login_" + UUID.randomUUID().toString();
    private static final String LOGIN3 = "login_" + UUID.randomUUID().toString();
    private static final String PASS = "pass_" + UUID.randomUUID().toString();
    private static final String PASS2 = "pass_" + UUID.randomUUID().toString();
    private static final String PASS3 = "pass_" + UUID.randomUUID().toString();
}
