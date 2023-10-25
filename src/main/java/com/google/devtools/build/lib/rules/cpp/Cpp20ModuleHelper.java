package com.google.devtools.build.lib.rules.cpp;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * to parse JSON content with format
 * <a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1689r5.html">p1689</a>
 * when discovering dependencies.
 * all related class are put here
 * for example,
 * <code>
 * {
 *   "revision": 0,
 *   "rules": [
 *     {
 *       "primary-output": "path/to/a.pcm",
 *       "provides": [
 *         {
 *           "is-interface": true,
 *           "logical-name": "a",
 *           "source-path": "path/to/a.cppm"
 *         }
 *       ],
 *       "requires": [
 *         {
 *           "logical-name": "b"
 *         }
 *       ]
 *     }
 *   ],
 *   "version": 1
 * }
 * </code>
 *
 *
 */
public class Cpp20ModuleHelper {
    static class Require {
        @SerializedName("logical-name")
        private String logicalName;

        @SerializedName("source-path")
        private String sourcePath;

        public String getLogicalName() {
            return logicalName;
        }

        public void setLogicalName(String logicalName) {
            this.logicalName = logicalName;
        }

        public String getSourcePath() {
            return sourcePath;
        }

        public void setSourcePath(String sourcePath) {
            this.sourcePath = sourcePath;
        }
    }

    static class Provide {
        @SerializedName("is-interface")
        private Boolean interfaceModule;
        @SerializedName("logical-name")
        private String logicalName;
        @SerializedName("source-path")
        private String sourcePath;

        @Override
        public String toString() {
            Gson gson = new Gson();
            return gson.toJson(this);
        }

        public Boolean getInterfaceModule() {
            return interfaceModule;
        }

        public void setInterfaceModule(Boolean interfaceModule) {
            this.interfaceModule = interfaceModule;
        }

        public String getLogicalName() {
            return logicalName;
        }

        public void setLogicalName(String logicalName) {
            this.logicalName = logicalName;
        }

        public String getSourcePath() {
            return sourcePath;
        }

        public void setSourcePath(String sourcePath) {
            this.sourcePath = sourcePath;
        }
    }

    static class Rule {
        @SerializedName("primary-output")
        private String primaryOutput;
        private List<Provide> provides;
        private List<Require> requires;

        @Override
        public String toString() {
            Gson gson = new Gson();
            return gson.toJson(this);
        }

        public String getPrimaryOutput() {
            return primaryOutput;
        }

        public void setPrimaryOutput(String primaryOutput) {
            this.primaryOutput = primaryOutput;
        }

        public List<Provide> getProvides() {
            return provides;
        }

        public void setProvides(List<Provide> provides) {
            this.provides = provides;
        }

        public List<Require> getRequires() {
            return requires;
        }

        public void setRequires(List<Require> requires) {
            this.requires = requires;
        }
    }

    static class Cpp20ModuleScanDepsResult {
        private Integer revision;
        private String version;
        private List<Rule> rules;

        @Override
        public String toString() {
            Gson gson = new Gson();
            return gson.toJson(this);
        }

        public Integer getRevision() {
            return revision;
        }

        public void setRevision(Integer revision) {
            this.revision = revision;
        }

        public String getVersion() {
            return version;
        }

        public void setVersion(String version) {
            this.version = version;
        }

        public List<Rule> getRules() {
            return rules;
        }

        public void setRules(List<Rule> rules) {
            this.rules = rules;
        }
    }

    static class ModuleDep {
        private boolean needProduceBMI = false;
        private String moduleName;
        private List<String> requireModules;

        public boolean isNeedProduceBMI() {
            return needProduceBMI;
        }

        public void setNeedProduceBMI(boolean needProduceBMI) {
            this.needProduceBMI = needProduceBMI;
        }

        public String getModuleName() {
            return moduleName;
        }

        public void setModuleName(String moduleName) {
            this.moduleName = moduleName;
        }

        public List<String> getRequireModules() {
            return requireModules;
        }

        public void setRequireModules(List<String> requireModules) {
            this.requireModules = requireModules;
        }
    }

    static class Cpp20ModulesInfo {
        Cpp20ModulesInfo() {
            modules = new HashMap<>();
            usages = new HashMap<>();
        }
        void merge(Cpp20ModulesInfo other) {
            this.modules.putAll(other.modules);
            this.usages.putAll(other.usages);
        }
        private final Map<String, String> modules;
        private final Map<String, List<String>> usages;

        public List<String> getRequireModules(String moduleName) {
            return usages.getOrDefault(moduleName, List.of());
        }
        public void addRequireModule(String moduleName, String requireModuleName) {
            usages.computeIfAbsent(moduleName, it -> new ArrayList<>());
            usages.get(moduleName).add(requireModuleName);
        }

        public String getModulePath(String moduleName) {
            return modules.getOrDefault(moduleName, null);
        }
        public void putModulePath(String moduleName, String modulePath) {
            modules.put(moduleName, modulePath);
        }
        static Cpp20ModulesInfo fromJSON(String json) {
            Gson gson = new Gson();
            return gson.fromJson(json, Cpp20ModulesInfo.class);
        }
    }
    static ModuleDep parseScanResult(String out) {
        Gson gson = new Gson();
        Cpp20ModuleScanDepsResult dep = gson.fromJson(out, Cpp20ModuleScanDepsResult.class);
        if (dep == null) {
            throw new RuntimeException("call clang-scan-deps error");
        }
        if (dep.getRules() == null) {
            throw new RuntimeException("call clang-scan-deps error");
        }
        if (dep.getRules().size() != 1) {
            throw new RuntimeException("expect only 1 rule, but got " + dep.getRules());
        }
        ModuleDep moduleDep = new ModuleDep();
        Rule rule = dep.getRules().get(0);
        if (rule.getProvides() != null && !rule.getProvides().isEmpty()) {
            if (rule.getProvides().size() != 1) {
                throw new RuntimeException("expect only 1 rule, but got " + rule.getProvides());
            }
            Provide provide = rule.getProvides().get(0);
            moduleDep.setModuleName(provide.getLogicalName());
            moduleDep.setNeedProduceBMI(true);
        }
        if (rule.getRequires() != null) {
            var requireModules = new ArrayList<String>(rule.getRequires().size());
            for (int i = 0; i < rule.getRequires().size(); i++) {
                Require require = rule.getRequires().get(i);
                requireModules.add(require.getLogicalName());
            }
            moduleDep.setRequireModules(requireModules);
        }
        return moduleDep;
    }
    public static boolean isCpp20ModuleCompilationAction(String actionName) {
        return CppActionNames.CPP20_MODULE_COMPILE.equals(actionName)
                || CppActionNames.CPP20_MODULE_CODEGEN.equals(actionName);
    }
}
