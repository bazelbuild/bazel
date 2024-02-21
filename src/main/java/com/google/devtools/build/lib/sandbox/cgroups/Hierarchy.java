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
public abstract class Hierarchy {
    public abstract Integer id();
    public abstract List<String> controllers();
    public abstract Path path();
    public boolean isV2() {
        return controllers().size() == 1 && controllers().contains("") && id() == 0;
    }

    /**
     * A regexp that matches entries in {@code /proc/self/cgroup}.
     *
     * The format is documented in https://man7.org/linux/man-pages/man7/cgroups.7.html
     */
    private static final Pattern PROC_CGROUPS_PATTERN =
        Pattern.compile("^(?<id>\\d+):(?<controllers>[^:]*):(?<file>.+)");

    static Hierarchy create(Integer id, List<String> controllers, Path path) {
        return new AutoValue_Hierarchy(id, controllers, path);
    }

    static List<Hierarchy> parse(File procCgroup) throws IOException {
        ImmutableList.Builder<Hierarchy> hierarchies = ImmutableList.builder();
        for (String line : Files.readLines(procCgroup, StandardCharsets.UTF_8)) {
            Matcher m = PROC_CGROUPS_PATTERN.matcher(line);
            if (!m.matches()) {
                continue;
            }

            Integer id = Integer.parseInt(m.group("id"));
            String path = m.group("file");
            List<String> cs = ImmutableList.copyOf(m.group("controllers").split(","));
            hierarchies.add(Hierarchy.create(id, cs, Paths.get(path)));
        }
        return hierarchies.build();
    }
}