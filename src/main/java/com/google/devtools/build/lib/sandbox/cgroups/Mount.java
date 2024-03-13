package com.google.devtools.build.lib.sandbox.cgroups;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@AutoValue
public abstract class Mount {
    /**
     * A regexp that matches cgroups entries in {@code /proc/mounts}.
     *
     * The format is documented in https://man7.org/linux/man-pages/man5/fstab.5.html
     */
    private static final Pattern CGROUPS_MOUNT_PATTERN =
        Pattern.compile("^[^\\s#]\\S*\\s+(?<file>\\S*)\\s+(?<vfstype>cgroup2?)\\s+(?<mntops>\\S*).*");

    public abstract Path path();
    public abstract String type();
    /**
     * Mount point options for this mount. In the context of this cgroup, this will
     * contain the controllers that are mounted at this mount point.
     */
    public abstract List<String> opts();
    public boolean isV2() {
        return type().equals("cgroup2");
    }

    static Mount create(Path path, String type, List<String> opts) {
        return new AutoValue_Mount(path, type, opts);
    }

    /**
     * Parses the cgroup mounts from the provided file and returns a list of mounts.
     * @param procMounts: a file containing the cgroup mounts, typically {@code /proc/mounts}.
     * @return The list of cgroup mounts in the file.
     */
    static List<Mount> parse(File procMounts) throws IOException {
        ImmutableList.Builder<Mount> mounts = ImmutableList.builder();

        for (String mount: Files.readLines(procMounts, StandardCharsets.UTF_8)) {
            Matcher m = CGROUPS_MOUNT_PATTERN.matcher(mount);
            if (!m.matches()) {
                continue;
            }

            String path = m.group("file");
            String type = m.group("vfstype");
            List<String> opts = ImmutableList.copyOf(m.group("mntops").split(","));
            mounts.add(Mount.create(Paths.get(path), type, opts));
        }
        return mounts.build();
    }
}