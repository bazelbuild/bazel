/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.configuration;


import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * This class can be injected in applications to log information about reflection
 * being used in the application code, and suggest appropriate ProGuard rules for
 * keeping the reflected classes, methods and/or fields.
 *
 * @author Johan Leys
 */
public class ConfigurationLogger implements Runnable
{
    public static final boolean LOG_ONCE = true;

    private static final String LOG_TAG = "ProGuard";

    public static final String CLASS_MAP_FILENAME = "classmap.txt";

    private static final String EMPTY_LINE = "\u00a0\n";

    // Set with missing class names.
    private static final Set<String> missingClasses = new HashSet<String>();

    // Map from class name to missing constructors.
    private static final Map<String, Set<MethodSignature>> missingConstructors = new HashMap<String, Set<MethodSignature>>();
    // Set of classes on which getConstructors or getDeclaredConstructors is invoked.
    private static final Set<String>                       constructorListingClasses = new HashSet<String>();

    // Map from class name to missing method signatures.
    private static final Map<String, Set<MethodSignature>> missingMethods       = new HashMap<String, Set<MethodSignature>>();
    // Set of classes on which getMethods or getDeclaredMethods is invoked.
    private static final Set<String>                       methodListingClasses = new HashSet<String>();

    // Map from class name to missing field names.
    private static final Map<String, Set<String>> missingFields       = new HashMap<String, Set<String>>();
    // Set of classes on which getFields or getDeclaredFields is invoked.
    private static final Set<String>              fieldListingCLasses = new HashSet<String>();

    // Map from obfuscated class name to original class name.
    private static Map<String, String> classNameMap;

    // Set of classes that have renamed or removed methods.
    private static Set<String>         classesWithObfuscatedMethods;

    // Set of classes that have renamed or removed fields.
    private static Set<String>         classesWithObfuscatedFields;

    private static Method logMethod;

    // Try to find the Android logging class.
    static
    {
        try
        {
            Class<?> logClass = Class.forName("android.util.Log");
            logMethod = logClass.getMethod("w", String.class, String. class);
        }
        catch (Exception e) {}
    }

    // Classes.

    /**
     * Log a failed call to Class.forName().
     *
     * @param callingClassName
     * @param missingClassName
     */
    public static void logForName(String callingClassName,
                                  String missingClassName)
    {
        logMissingClass(callingClassName, "Class", "forName", missingClassName);
    }

    /**
     * Log a failed call to ClassLoader.loadClass().
     *
     * @param callingClassName
     * @param missingClassName
     */
    public static void logLoadClass(String callingClassName,
                                  String missingClassName)
    {
        logMissingClass(callingClassName, "ClassLoader", "loadClass", missingClassName);
    }


    /**
     * Log a failed call to Class.forName().
     *
     * @param callingClassName
     * @param missingClassName
     */
    public static void logMissingClass(String callingClassName,
                                  String invokedClassName,
                                  String invokedMethodName,
                                  String missingClassName)
    {
        if (!LOG_ONCE || !missingClasses.contains(missingClassName))
        {
            missingClasses.add(missingClassName);
            log(
                "The class '" + originalClassName(callingClassName) + "' is calling " + invokedClassName + "." + invokedMethodName + " to retrieve\n" +
                "the class '" + missingClassName + "', but the latter could not be found.\n" +
                "It may have been obfuscated or shrunk.\n" +
                "You should consider preserving the class with its original name,\n" +
                "with a setting like:\n" +
                EMPTY_LINE +
                keepClassRule(missingClassName) + "\n" +
                EMPTY_LINE);
        }
    }


    // Constructors.


    /**
     * Log a failed call to Class.getDeclaredConstructor().
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param constructorParameters
     */
    public static void logGetDeclaredConstructor(String  invokingClassName,
                                                 Class   reflectedClass,
                                                 Class[] constructorParameters)
    {
        logGetConstructor(invokingClassName, "getDeclaredConstructor", reflectedClass, constructorParameters);
    }


    /**
     * Log a failed call to Class.getConstructor().
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param constructorParameters
     */
    public static void logGetConstructor(String  invokingClassName,
                                         Class   reflectedClass,
                                         Class[] constructorParameters)
    {
        logGetConstructor(invokingClassName, "getConstructor", reflectedClass, constructorParameters);
    }


    /**
     * Log a failed call to one of the constructor retrieving methods on Class.
     *
     * @param invokingClassName
     * @param invokedMethodName
     * @param reflectedClass
     * @param constructorParameters
     */
    public static void logGetConstructor(String  invokingClassName,
                                         String  invokedMethodName,
                                         Class   reflectedClass,
                                         Class[] constructorParameters)
    {
        MethodSignature signature = new MethodSignature("<init>", constructorParameters);

        Set<MethodSignature> constructors = missingConstructors.get(reflectedClass.getName());
        if (constructors == null)
        {
            constructors = new HashSet<MethodSignature>();
            missingConstructors.put(reflectedClass.getName(), constructors);
        }

        if ((!LOG_ONCE || !constructors.contains(signature)) && !isLibraryClass(reflectedClass))
        {
            constructors.add(signature);
            log(
                "The class '" + originalClassName(invokingClassName) + "' is calling Class." + invokedMethodName + "\n" +
                "on class '" + originalClassName(reflectedClass) + "' to retrieve\n" +
                "the constructor with signature (" + originalSignature(signature) + "), but the latter could not be found.\n" +
                "It may have been obfuscated or shrunk.\n" +
                "You should consider preserving the constructor, with a setting like:\n" +
                EMPTY_LINE +
                keepConstructorRule(reflectedClass.getName(), signature) + "\n" +
                EMPTY_LINE);
        }
    }


    /**
     * Log a call to Class.getDeclaredConstructors().
     *
     * @param invokingClassName
     * @param reflectedClass
     */
    public static void logGetDeclaredConstructors(String invokingClassName,
                                                  Class  reflectedClass    )
    {
        logGetConstructors(invokingClassName, reflectedClass, "getDeclaredConstructors");
    }


    /**
     * Log a call to Class.getConstructors().
     *
     * @param invokingClassName
     * @param reflectedClass
     */
    public static void logGetConstructors(String invokingClassName,
                                          Class  reflectedClass    )
    {
        logGetConstructors(invokingClassName, reflectedClass, "getConstructors");
    }


    /**
     * Log a call to one of the constructor listing methods on Class.
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param reflectedMethodName
     */
    private static void logGetConstructors(String invokingClassName,
                                           Class  reflectedClass,
                                           String reflectedMethodName)
    {
        initializeMappings();
        if (classesWithObfuscatedMethods.contains(reflectedClass.getName()) &&
            !constructorListingClasses.contains(reflectedClass.getName()) &&
            !isLibraryClass(reflectedClass))
        {
            constructorListingClasses.add(reflectedClass.getName());
            log(
                "The class '" + originalClassName(invokingClassName) + "' is calling Class." + reflectedMethodName + "\n" +
                "on class '" + originalClassName(reflectedClass) + "' to retrieve its constructors.\n" +
                "You might consider preserving all constructors with their original names,\n" +
                "with a setting like:\n" +
                EMPTY_LINE +
                keepAllConstructorsRule(reflectedClass) + "\n" +
                EMPTY_LINE);
        }
    }


    // Methods.


    /**
     * Log a failed call to Class.getDeclaredMethod().
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param reflectedMethodName
     * @param methodParameters
     */
    public static void logGetDeclaredMethod(String  invokingClassName,
                                            Class   reflectedClass,
                                            String  reflectedMethodName,
                                            Class[] methodParameters    )
    {
        logGetMethod(invokingClassName, "getDeclaredMethod", reflectedClass, reflectedMethodName, methodParameters);
    }


    /**
     * Log a failed call to Class.getMethod().
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param reflectedMethodName
     * @param methodParameters
     */
    public static void logGetMethod(String  invokingClassName,
                                    Class   reflectedClass,
                                    String  reflectedMethodName,
                                    Class[] methodParameters    )
    {
        logGetMethod(invokingClassName, "getMethod", reflectedClass, reflectedMethodName, methodParameters);
    }


    /**
     * Log a failed call to one of the method retrieving methods on Class.
     * @param invokingClassName
     * @param invokedReflectionMethodName
     * @param reflectedClass
     * @param reflectedMethodName
     * @param methodParameters
     */
    private static void logGetMethod(String  invokingClassName,
                                     String  invokedReflectionMethodName,
                                     Class   reflectedClass,
                                     String  reflectedMethodName,
                                     Class[] methodParameters    )
    {
        Set<MethodSignature> methods = missingMethods.get(reflectedClass.getName());
        if (methods == null)
        {
            methods = new HashSet<MethodSignature>();
            missingMethods.put(reflectedClass.getName(), methods);
        }

        MethodSignature signature = new MethodSignature(reflectedMethodName, methodParameters);
        if (!methods.contains(signature) && !isLibraryClass(reflectedClass))
        {
            methods.add(signature);
            log(
                "The class '" + originalClassName(invokingClassName) +
                "' is calling Class." + invokedReflectionMethodName + "\n" +
                "on class '" + originalClassName(reflectedClass) +
                "' to retrieve the method\n" +
                reflectedMethodName + "(" + originalSignature(signature) + "),\n" +
                "but the latter could not be found. It may have been obfuscated or shrunk.\n" +
                "You should consider preserving the method with its original name,\n" +
                "with a setting like:\n" +
                EMPTY_LINE +
                keepMethodRule(reflectedClass.getName(), reflectedMethodName, signature) + "\n" +
                EMPTY_LINE);
        }
    }


    /**
     * Log a call to Class.getDeclaredMethods().
     *
     * @param invokingClassName
     * @param reflectedClass
     */
    public static void logGetDeclaredMethods(String invokingClassName,
                                             Class  reflectedClass    )
    {
        logGetMethods(invokingClassName, "getDeclaredMethods", reflectedClass);
    }


    /**
     * Log a call to Class.getMethods().
     *
     * @param invokingClassName
     * @param reflectedClass
     */
    public static void logGetMethods(String invokingClassName,
                                     Class  reflectedClass    )
    {
        logGetMethods(invokingClassName, "getMethods", reflectedClass);
    }


    /**
     * Log a call to one of the method listing methods on Class.
     *
     * @param invokingClassName
     * @param invokedReflectionMethodName
     * @param reflectedClass
     */
    private static void logGetMethods(String invokingClassName,
                                      String invokedReflectionMethodName,
                                      Class  reflectedClass     )
    {
        initializeMappings();
        if (classesWithObfuscatedMethods.contains(reflectedClass.getName()) &&
            !methodListingClasses.contains(reflectedClass.getName()) &&
            !isLibraryClass(reflectedClass))
        {
            methodListingClasses.add(reflectedClass.getName());
            log(
                "The class '" + originalClassName(invokingClassName) +
                "' is calling Class." + invokedReflectionMethodName + "\n" +
                "on class '" + originalClassName(reflectedClass) +
                "' to retrieve its methods.\n" +
                "You might consider preserving all methods with their original names,\n" +
                "with a setting like:\n" +
                EMPTY_LINE +
                keepAllMethodsRule(reflectedClass) + "\n" +
                EMPTY_LINE);
        }
    }


    // Fields.


    /**
     * Log a failed call to Class.getField().
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param reflectedFieldName
     */
    public static void logGetField(String invokingClassName,
                                   Class  reflectedClass,
                                   String reflectedFieldName)
    {
        logGetField(invokingClassName, "getField", reflectedClass, reflectedFieldName);
    }


    /**
     * Log a failed call to Class.getDeclaredField().
     *
     * @param invokingClassName
     * @param reflectedClass
     * @param reflectedFieldName
     */
    public static void logGetDeclaredField(String invokingClassName,
                                           Class  reflectedClass,
                                           String reflectedFieldName)
    {
        logGetField(invokingClassName, "getDeclaredField", reflectedClass, reflectedFieldName);
    }


    /**
     * Log a failed call to one of the field retrieving methods of Class.
     *
     * @param invokingClassName
     * @param invokedReflectionMethodName
     * @param reflectedClass
     * @param reflectedFieldName
     */
    private static void logGetField(String invokingClassName,
                                    String invokedReflectionMethodName,
                                    Class  reflectedClass,
                                    String reflectedFieldName )
    {
        Set<String> fields = missingFields.get(reflectedClass.getName());
        if (fields == null)
        {
            fields = new HashSet<String>();
            missingFields.put(reflectedClass.getName(), fields);
        }

        if ((!LOG_ONCE || !fields.contains(reflectedFieldName)) &&
            !isLibraryClass(reflectedClass))
        {
            fields.add(reflectedFieldName);
            log(
                "The class '" + originalClassName(invokingClassName) +
                "' is calling Class." + invokedReflectionMethodName + "\n" +
                "on class '" + originalClassName(reflectedClass) +
                "' to retrieve the field '" + reflectedFieldName + "',\n" +
                "but the latter could not be found. It may have been obfuscated or shrunk.\n" +
                "You should consider preserving the field with its original name,\n" +
                "with a setting like:\n" +
                EMPTY_LINE +
                keepFieldRule(reflectedClass.getName(), reflectedFieldName) + "\n" +
                EMPTY_LINE);
        }
    }


    /**
     * Log a call to Class.getDeclaredFields().
     *
     * @param invokingClassName
     * @param reflectedClass
     */
    public static void logGetDeclaredFields(String invokingClassName,
                                            Class  reflectedClass    )
    {
        logGetFields(invokingClassName, "getDeclaredFields", reflectedClass);
    }


    /**
     * Log a call to Class.getFields().
     *
     * @param invokingClassName
     * @param reflectedClass
     */
    public static void logGetFields(String invokingClassName,
                                    Class  reflectedClass    )
    {
        logGetFields(invokingClassName, "getFields", reflectedClass);
    }


    /**
     * Log a call to one of the field listing methods on Class.
     *
     * @param invokingClassName
     * @param invokedReflectionMethodName
     * @param reflectedClass
     */
    private static void logGetFields(String invokingClassName,
                                     String invokedReflectionMethodName,
                                     Class  reflectedClass     )
    {
        initializeMappings();
        if (classesWithObfuscatedFields.contains(reflectedClass.getName()) &&
            !fieldListingCLasses.contains(reflectedClass.getName()) &&
            !isLibraryClass(reflectedClass))
        {
            fieldListingCLasses.add(reflectedClass.getName());
            log(
                "The class '" + originalClassName(invokingClassName) +
                "' is calling Class." + invokedReflectionMethodName + "\n" +
                "on class '" + originalClassName(reflectedClass) +
                "' to retrieve its fields.\n" +
                "You might consider preserving all fields with their original names,\n" +
                "with a setting like:\n" +
                EMPTY_LINE +
                keepAllFieldsRule(reflectedClass) + "\n" +
                EMPTY_LINE);
        }
    }


    // Implementations for Runnable.

    public void run()
    {
        printConfiguration();
    }


    private static void printConfiguration()
    {
        log("The following settings may help solving issues related to\n" +
            "missing classes, methods and/or fields:\n");

        for (String clazz : missingClasses)
        {
            log(keepClassRule(clazz) + "\n");
        }

        for (String clazz : missingConstructors.keySet())
        {
            for (MethodSignature constructor : missingConstructors.get(clazz))
            {
                log(keepConstructorRule(clazz, constructor) + "\n");
            }
        }

        for (String clazz : missingMethods.keySet())
        {
            for (MethodSignature method : missingMethods.get(clazz))
            {
                log(keepMethodRule(clazz, method.name, method) + "\n");
            }
        }

        for (String clazz : missingFields.keySet())
        {
            for (String field : missingFields.get(clazz))
            {
                log(keepFieldRule(clazz, field) + "\n");
            }
        }
    }


    // ProGuard rules.

    private static String keepClassRule(String className)
    {
        return "-keep class " + className;
    }


    private static String keepConstructorRule(String          className,
                                              MethodSignature constructorParameters)
    {
        return "-keepclassmembers class " + originalClassName(className) + " {\n" +
               "    public <init>(" + originalSignature(constructorParameters) + ");\n" +
               "}";
    }


    private static String keepMethodRule(String          className,
                                         String          methodName,
                                         MethodSignature constructorParameters)
    {
        return "-keepclassmembers class " + originalClassName(className) + " {\n" +
               "    *** " + methodName + "(" + originalSignature(constructorParameters) + ");\n" +
               "}";
    }


    private static String keepFieldRule(String className,
                                        String fieldName)
    {
        return "-keepclassmembers class " + originalClassName(className) + " {\n" +
               "    *** " + fieldName + ";\n" +
               "}";
    }


    private static String keepAllConstructorsRule(Class className)
    {
        return "-keepclassmembers class " + originalClassName(className) + " {\n" +
               "    <init>(...);\n" +
               "}";
    }


    private static String keepAllMethodsRule(Class className)
    {
        return "-keepclassmembers class " + originalClassName(className) + " {\n" +
               "    <methods>;\n" +
               "}";
    }


    private static String keepAllFieldsRule(Class className)
    {
        return "-keepclassmembers class " + originalClassName(className) + " {\n" +
               "    <fields>;\n" +
               "}";
    }


    private static String originalClassName(Class className)
    {
        return originalClassName(className.getName());
    }


    private static String originalClassName(String className)
    {
        initializeMappings();
        String originalClassName = classNameMap.get(className);
        return originalClassName != null ? originalClassName : className;
    }


    /**
     * Simple heuristic to see if the given class is a library class or not.
     *
     * @param clazz
     * @return
     */
    private static boolean isLibraryClass(Class clazz)
    {
        return clazz.getClassLoader() == String.class.getClassLoader();
    }


    /**
     * Log a message, either on the Android Logcat, if available, or on the
     * Standard error outputstream otherwise.
     *
     * @param message the message to be logged.
     */
    private static void log(String message)
    {
        if (logMethod != null)
        {
            try
            {
                logMethod.invoke(null, LOG_TAG, message);
            }
            catch (Exception e)
            {
                System.err.println(message);
            }
        }
        else
        {
            System.err.println(message);
        }
    }


    private  static void initializeMappings()
    {
        if (classNameMap == null)
        {
            classNameMap                 = new HashMap<String, String> ();
            classesWithObfuscatedMethods = new HashSet<String>         ();
            classesWithObfuscatedFields  = new HashSet<String>         ();

            String line;
            try
            {
                BufferedReader reader =
                    new BufferedReader(
                        new InputStreamReader(
                            ConfigurationLogger.class.getClassLoader().getResourceAsStream(CLASS_MAP_FILENAME)));

                while ((line = reader.readLine()) != null)
                {
                    StringTokenizer tokenizer            = new StringTokenizer(line, ",");
                    String          originalClassName    = tokenizer.nextToken();
                    String          obfuscatedClassName  = tokenizer.nextToken();
                    boolean         hasObfuscatedMethods = tokenizer.nextToken().equals("1");
                    boolean         hasObfuscatedFields  = tokenizer.nextToken().equals("1");

                    classNameMap.put(obfuscatedClassName, originalClassName);

                    if (hasObfuscatedMethods)
                    {
                        classesWithObfuscatedMethods.add(obfuscatedClassName);
                    }

                    if (hasObfuscatedFields)
                    {
                        classesWithObfuscatedFields.add(obfuscatedClassName);
                    }
                }
                reader.close();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }


    private static String originalSignature(MethodSignature signature)
    {
        StringBuilder stringBuilder = new StringBuilder();
        boolean       first         = true;
        for (String clazz : signature.parameters)
        {
            if (first)
            {
                first = false;
            }
            else
            {
                stringBuilder.append(",");
            }
            stringBuilder.append(originalClassName(clazz));
        }
        return stringBuilder.toString();
    }


    public static class MethodSignature
    {
        private String   name;
        private String[] parameters;


        public MethodSignature(String name, Class[] parameters)
        {
            this.name       = name;
            this.parameters = new String[parameters.length];
            for (int i = 0; i < parameters.length; i++)
            {
                this.parameters[i] = parameters[i].getName();
            }
        }


        // Implementations for Object.

        public boolean equals(Object o)
        {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            MethodSignature that = (MethodSignature)o;

            if (!name.equals(that.name)) return false;
            return Arrays.equals(parameters, that.parameters);
        }


        public int hashCode()
        {
            int result = name.hashCode();
            result = 31 * result + Arrays.hashCode(parameters);
            return result;
        }
    }
}
