package com.google.devtools.build.lib.bazel.repository.downloader;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NetrcCredentialsProvider implements CredentialsProvider {

    private Path netrcFilePath;

    private String netrcFileContent;

    private static final Pattern NETRC_PATTERN = Pattern
            .compile("(^\\s*machine|.*machine)\\s+(\\S+)(\\s+login\\s+(\\S+))?(\\s+password\\s+(\\S+))?.*");

    private static final int MACHINE_GROUP_IDX = 2;
    private static final int LOGIN_GROUP_IDX = 4;
    private static final int PASSWORD_GROUP_IDX = 6;

    private static final Map<Path, NetrcCredentialsProvider> instances = new HashMap<>();

    public static NetrcCredentialsProvider getInstance(Path netrcFilePath) {
        if (instances.get(netrcFilePath) != null) {
            return instances.get(netrcFilePath);
        }

        synchronized (instances) {
            if (instances.get(netrcFilePath) == null) {
                instances.put(netrcFilePath, new NetrcCredentialsProvider(netrcFilePath));
            }
        }

        return instances.get(netrcFilePath);
    }

    NetrcCredentialsProvider(Path netrcFilePath) {
        this.netrcFilePath = netrcFilePath;
    }

    @Override
    public Optional<Credentials> getCredentials(String host) {
        return Optional.of(fetchNetrcFileContentIfNeeded())
                .map(NETRC_PATTERN::matcher)
                .filter(m -> findHost(m, host))
                .map(m -> new Credentials(m.group(LOGIN_GROUP_IDX), m.group(PASSWORD_GROUP_IDX)));
    }

    private String fetchNetrcFileContentIfNeeded() {
        if (netrcFileContent == null) {
            synchronized (this) {
                if (netrcFileContent == null) {
                    if (Files.notExists(netrcFilePath))
                        throw new NetrcCredentialsProviderException(String.format(".netrc file by path '%s' does not exist", netrcFilePath.toString()));

                    try {
                        netrcFileContent = new String(Files.readAllBytes(netrcFilePath), StandardCharsets.UTF_8).trim();
                    } catch (IOException e) {
                        throw new NetrcCredentialsProviderException(e.getMessage(), e);
                    }
                }
            }
        }
        return netrcFileContent;
    }

    private boolean findHost(Matcher matcher, String host) {
        while (matcher.find()) {
            if (host.endsWith(matcher.group(MACHINE_GROUP_IDX))) {
                return true;
            }
        }
        return false;
    }
}
