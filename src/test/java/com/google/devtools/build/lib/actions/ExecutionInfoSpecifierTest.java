package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class ExecutionInfoSpecifierTest extends BuildViewTestCase {

    // for all subclasses
    //      create an instance
    //      call addExecutionInfo
    //      compare with initial instance
    //          deep comparison by fields (reflection)
    //      if any of the fields are different
    //          -> FAIL with message "It seems that you've added a new field <filed name> to the action <type here>. This field is not set properly in the constructor used for addExecutionInfo method. Please add it there"

    @Test
    public void test_super_puper_test() throws ClassNotFoundException, NoSuchMethodException, IllegalAccessException, InvocationTargetException, InstantiationException, NoSuchFieldException {
        List<Class<?>> subclasses = new ArrayList<>();
        subclasses.add(Class.forName("com.google.devtools.build.lib.rules.cpp.CppCompileAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.rules.cpp.CppLinkAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.rules.cpp.FakeCppCompileAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.rules.cpp.LtoBackendAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.rules.genrule.GenRuleAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.analysis.actions.SpawnAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.analysis.actions.StarlarkAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.analysis.extra.ExtraAction"));
        subclasses.add(Class.forName("com.google.devtools.build.lib.analysis.test.TestRunnerAction"));

        for (Class<?> action : subclasses) {
//            ExecutionInfoSpecifier instance = (ExecutionInfoSpecifier) action.newInstance();
// Does not work, becasue there are no null constructors.  => will create the full instances and use reflection to do a deep comparison
//             => put it into one class here or per-action test?
            ExecutionInfoSpecifier instance = (ExecutionInfoSpecifier) action.newInstance();
            ActionAnalysisMetadata modified = instance.addExecutionInfo(ImmutableMap.of("one", "two"));

           assertThat(modified).isInstanceOf(instance.getClass());
           Field[] fields = instance.getClass().getDeclaredFields();
           for (Field f : fields){
               assertThat(f.equals(modified.getClass().getField(f.getName()))).isTrue();
           }
        }

    }

}